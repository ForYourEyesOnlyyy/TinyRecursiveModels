import os
import math
import random
from itertools import islice
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import pydantic
from tqdm import tqdm
import wandb

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from utils.wandb import build_run_name, build_arch_tags
from models.losses import IGNORE_LABEL_ID
from models.losses import RTifyLossHead
from models.recursive_reasoning.TRM_RTify import TRM_RtifyCarry
from models.recursive_reasoning.TRM_NoACT import TRMCarry

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    arch: ArchConfig

    project_name: Optional[str] = None
    run_name: Optional[str] = None

    data_paths: List[str]
    data_split_train: str
    data_split_eval: str
    blank_id: int
    global_batch_size: int
    max_eval_batches: Optional[int] = None

    lr: float
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999

    seed: int = 0
    epochs: int
    eval_interval: int
    device: str = "auto"  # "cpu" | "cuda" | "mps" | "auto"

    checkpoint_dir: Optional[str] = None
    save_every: Optional[int] = None
    save_best: bool = True
    resume: bool = False
    load_checkpoint: Optional[str] = None
    best_metric: str = "eval_acc_blank"
    best_mode: str = "max"  # "max" or "min"

    wandb: dict[str, Any] = {}


def load_synced_config(hydra_cfg: DictConfig) -> TrainConfig:
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg = TrainConfig(**cfg_dict)

    if cfg.project_name is None:
        proj = cfg.wandb.get("project") if cfg.wandb else None
        if proj is None:
            proj = cfg.arch.name + "-" + os.path.basename(cfg.data_paths[0]).capitalize()
        cfg.project_name = proj

    if cfg.run_name is None:
        cfg.run_name = build_run_name(cfg)

    if cfg.checkpoint_dir is None:
        cfg.checkpoint_dir = os.path.join("checkpoints", cfg.project_name, cfg.run_name)

    if cfg.save_every is None:
        cfg.save_every = cfg.eval_interval

    return cfg



def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ

def get_rank_world() -> tuple[int, int, int]:
    if not is_distributed():
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size, local_rank


def setup_distributed(cfg: TrainConfig) -> tuple[int, int, int, torch.device]:
    rank, world_size, local_rank = get_rank_world()

    if is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if cfg.device in ("cuda", "auto") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed() else "cuda")
    else:
        device = get_device(cfg)

    return rank, world_size, local_rank, device


def broadcast_model(model: nn.Module, src: int = 0):
    if not dist.is_initialized():
        return
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t, src=src)


def broadcast_objects(obj_list: list, src: int = 0):
    if not dist.is_initialized():
        return obj_list
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list


def allreduce_grads(model: nn.Module):
    """Average gradients across ranks (manual DDP), safe if some grads are None."""
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    for p in model.parameters():
        if not p.requires_grad:
            continue

        if p.grad is None:
            grad = torch.zeros_like(p, device=p.device, dtype=p.dtype)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            continue

        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)
        

def ddp_sum_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """
    Sum-reduce a dict of scalar metrics across ranks.

    - If dist is not initialized, returns input unchanged.
    - Uses float64 for numeric stability.
    - Reduction is SUM; caller handles normalization (divide) afterwards.

    Keys are reduced in sorted order to ensure consistent packing.
    """
    if not dist.is_initialized():
        return metrics

    keys = sorted(metrics.keys())
    vec = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    return {k: float(vec[i].item()) for i, k in enumerate(keys)}



def make_dataloader(cfg: TrainConfig, split: str, rank: int, world_size: int):
    ds_cfg = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=1,
        rank=rank,
        num_replicas=world_size,
    )
    dataset = PuzzleDataset(ds_cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=True,
    )
    return loader, dataset.metadata


def build_model_from_cfg(cfg: TrainConfig, metadata, device: torch.device) -> nn.Module:
    model_cls = load_model_class(cfg.arch.name)

    arch_dict = cfg.arch.model_dump()
    for k in ("name", "seq_len", "vocab_size", "num_puzzle_identifiers"):
        arch_dict.pop(k, None)

    model_cfg_dict = {
        **arch_dict,
        "seq_len": int(metadata.seq_len),
        "vocab_size": int(metadata.vocab_size),
    }

    model = model_cls(model_cfg_dict).to(device)
    return model



def _resolve_ckpt_dir(cfg: TrainConfig) -> str:
    base = cfg.checkpoint_dir
    os.makedirs(base, exist_ok=True)
    return base


def save_checkpoint(path: str, model, opt, epoch: int, step: int, cfg: TrainConfig, best_score=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict() if opt is not None else None,
        "epoch": epoch,
        "step": step,
        "best_score": best_score,
        "config": cfg.model_dump(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }
    if torch.cuda.is_available():
        ckpt["rng_torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, opt, device: torch.device):
    if not (path and os.path.isfile(path)):
        print(f"[checkpoint] no checkpoint at: {path}")
        return 0, 0, None, None

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[checkpoint] model strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    if opt is not None and ckpt.get("optimizer_state") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[checkpoint] optimizer state not loaded: {e}")

    try:
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])

        rng = ckpt["rng_torch"]
        if isinstance(rng, np.ndarray):
            rng = torch.from_numpy(rng)
        rng = rng.to("cpu")
        if rng.dtype != torch.uint8:
            rng = rng.to(torch.uint8)
        torch.random.set_rng_state(rng)

        if "rng_torch_cuda" in ckpt and torch.cuda.is_available():
            states = []
            for s in ckpt["rng_torch_cuda"]:
                if isinstance(s, np.ndarray):
                    s = torch.from_numpy(s)
                s = s.to("cpu")
                if s.dtype != torch.uint8:
                    s = s.to(torch.uint8)
                states.append(s)
            torch.cuda.set_rng_state_all(states)

    except Exception as e:
        print(f"[checkpoint] RNG restore failed: {e}")

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    best_score = ckpt.get("best_score", None)
    wandb_run_id = ckpt.get("wandb_run_id", None)

    print(f"[checkpoint] loaded: {path} (epoch={epoch}, step={step}, best={best_score}, id={wandb_run_id})")
    return epoch, step, best_score, wandb_run_id



def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step, base_lr, warmup, total_steps, min_ratio):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if min_ratio >= 0.9999:
        return base_lr
    t = (step - warmup) / max(1, total_steps - warmup)
    return min_ratio * base_lr + 0.5 * (1 - min_ratio) * base_lr * (1 + math.cos(math.pi * t))


def print_gpu_stats(device: torch.device, rank: int, world_size: int):
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    alloc_gb = torch.cuda.memory_allocated(idx) / (1024**3)
    reserv_gb = torch.cuda.memory_reserved(idx) / (1024**3)
    print(f"[rank {rank}/{world_size}] GPU {idx}: {name} | total={total_gb:.1f}GB alloc={alloc_gb:.2f}GB reserved={reserv_gb:.2f}GB")


@torch.no_grad()
def global_accuracy_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_LABEL_ID,
) -> Tuple[int, int]:
    """
    Returns (correct, total) over all valid tokens.
    """
    preds = logits.argmax(dim=-1)
    valid = (labels != ignore_index)
    correct = ((preds == labels) & valid).sum().item()
    total = valid.sum().item()
    return int(correct), int(total)


@torch.no_grad()
def blank_accuracy_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    *,
    blank_id: int,
    ignore_index: int = IGNORE_LABEL_ID,
) -> Tuple[int, int]:
    """
    Returns (correct, total) restricted to blank positions of the input.
    """
    preds = logits.argmax(dim=-1)
    blank_mask = (inputs == blank_id)
    valid = (labels != ignore_index) & blank_mask
    correct = ((preds == labels) & valid).sum().item()
    total = valid.sum().item()
    return int(correct), int(total)


@torch.no_grad()
def exact_accuracy_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_LABEL_ID,
) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)                # [B, L]
    mask = (labels != ignore_index)              # [B, L]

    loss_counts = mask.sum(-1)                   # [B]
    valid_metrics = (loss_counts > 0)            # [B]

    is_correct = mask & (preds == labels)        # [B, L]
    seq_is_correct = (is_correct.sum(-1) == loss_counts)  # [B]

    correct_exact = (valid_metrics & seq_is_correct).sum().item()
    total_exact = valid_metrics.sum().item()

    return int(correct_exact), int(total_exact)

# ------------------------------------------------------------
# Train epoch
# ------------------------------------------------------------

def train_one_rtify_step(
    loss_head,
    model,
    batch,
    carry,
    opt,
    *,
    global_batch_size,
    global_step
):
    in_warmup = global_step < model.config.halt_warmup_steps
 
    new_carry, loss, metrics, _, all_finish = loss_head(
        carry=carry,
        batch={**batch, "force_fixed_steps": in_warmup},
        return_keys=(),
    )
 
    opt.zero_grad(set_to_none=True)
 
    (loss / global_batch_size).backward()
 
    allreduce_grads(model)
 
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 
    opt.step()
 
    new_carry = TRM_RtifyCarry(
        inner_carry=TRMCarry(
            Z_S=new_carry.inner_carry.Z_S.detach(),
            Z_R=new_carry.inner_carry.Z_R.detach(),
        ),
        steps=new_carry.steps,
        halted=new_carry.halted,
        last_logits=new_carry.last_logits.detach(),
        phi=new_carry.phi.detach(),
    )
 
    return new_carry, loss, metrics, all_finish
 
 
# ------------------------------------------------------------
# Train epoch
# ------------------------------------------------------------
 
def train_one_epoch(
    model,
    loss_head,
    loader,
    device,
    *,
    base_lr,
    warmup,
    total_steps,
    min_ratio,
    step0,
    opt,
    use_wandb,
    epoch,
    global_batch_size,
    rank,
):
    model.train()
    step = step0
 
    for _, batch, _ in loader:
 
        batch  = {k: v.to(device).long() for k, v in batch.items()}
        labels = batch["labels"]
        carry  = model.initial_carry(batch)
 
        sum_lm_loss      = 0.0
        sum_halt_penalty = 0.0
        sum_readiness    = 0.0
        sum_evidence     = 0.0
        sum_active       = 0.0
        n_steps          = 0
        last_theta       = None
 
        while True:
 
            lr_now = cosine_warmup_lr(step, base_lr, warmup, total_steps, min_ratio)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
 
            carry, loss, metrics, all_finish = train_one_rtify_step(
                loss_head,
                model,
                batch,
                carry,
                opt,
                global_batch_size=global_batch_size,
                global_step=epoch + 1,         
            )
 
            sum_lm_loss      += metrics["lm_loss"].item()
            sum_halt_penalty += metrics["halt_penalty"].item()
            sum_readiness    += metrics["readiness"].item()
            sum_evidence     += metrics["evidence_sum"].item()
            sum_active       += metrics["active_count"].item()
            n_steps          += 1
            last_theta        = metrics["theta"].item()
 
            step += 1
 
            local_done = carry.halted.all()
            if dist.is_initialized():
                flag = torch.tensor([int(local_done)], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                global_done = bool(flag.item())
            else:
                global_done = bool(local_done)
 
            if global_done:
                break
 
        final_logits = carry.last_logits          # [B, L, V]
        steps_mean   = carry.steps.float().mean().item()
        phi_mean     = carry.phi.float().mean().item()
 
        c, t      = global_accuracy_counts(final_logits, labels, ignore_index=IGNORE_LABEL_ID)
        acc_all   = c / max(t, 1)
 
        c, t      = exact_accuracy_counts(final_logits, labels, ignore_index=IGNORE_LABEL_ID)
        acc_exact = c / max(t, 1)
 
        lm_loss_mean      = sum_lm_loss      / max(n_steps, 1)
        halt_penalty_mean = sum_halt_penalty / max(n_steps, 1)
        readiness_mean    = sum_readiness    / max(n_steps, 1)
        evidence_mean     = sum_evidence     / max(sum_active, 1.0)
 
        if rank == 0 and use_wandb:
            wandb.log({
                "train/accuracy":       acc_all,
                "train/exact_accuracy": acc_exact,
                "train/steps_mean":     steps_mean,
                "train/phi_mean":       phi_mean,
 
                "train/lm_loss":        lm_loss_mean      / global_batch_size,
                "train/halt_penalty":   halt_penalty_mean / global_batch_size,
                "train/readiness":      readiness_mean    / global_batch_size,
                "train/evidence_mean":  evidence_mean,
 
                # Scalar
                "train/theta":          last_theta,
 
                "epoch":       epoch + 1,
                "global_step": step,
            })
 
    return step
 
 
# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
 
@torch.no_grad()
def evaluate(
    model,
    loss_head,
    loader,
    device,
    *,
    max_batches,
    rank,
):
    loss_head.eval()
 
    sum_lm_loss      = 0.0
    sum_halt_penalty = 0.0
    sum_evidence     = 0.0
    sum_active       = 0.0
    sum_total_loss   = 0.0
    n_steps_total    = 0
 
    sum_acc          = 0.0
    sum_exact        = 0.0
    sum_steps        = 0.0
    sum_phi          = 0.0
    n_samples        = 0
 
    iterable = loader if max_batches is None else islice(loader, max_batches)
 
    for _, batch, _ in iterable:
 
        batch  = {k: v.to(device).long() for k, v in batch.items()}
        labels = batch["labels"]
        carry  = model.initial_carry(batch)
 
        ep_lm_loss      = 0.0
        ep_halt_penalty = 0.0
        ep_evidence     = 0.0
        ep_active       = 0.0
        ep_total_loss   = 0.0
        ep_steps        = 0
 
        while True:
            carry, loss, metrics, _, all_finish = loss_head(
                carry=carry,
                batch=batch,
                return_keys=[],
            )
 
            ep_lm_loss      += metrics["lm_loss"].item()
            ep_halt_penalty += metrics["halt_penalty"].item()
            ep_evidence     += metrics["evidence_sum"].item()
            ep_active       += metrics["active_count"].item()
            ep_total_loss   += loss.item()
            ep_steps        += 1
 
            if bool(all_finish):
                break
 
        sum_lm_loss      += ep_lm_loss
        sum_halt_penalty += ep_halt_penalty
        sum_evidence     += ep_evidence
        sum_active       += ep_active
        sum_total_loss   += ep_total_loss
        n_steps_total    += ep_steps
 
        final_logits = carry.last_logits          # [B, L, V]
        B            = final_logits.shape[0]
 
        c, t  = global_accuracy_counts(final_logits, labels, ignore_index=IGNORE_LABEL_ID)
        sum_acc   += c
 
        c, t  = exact_accuracy_counts(final_logits, labels, ignore_index=IGNORE_LABEL_ID)
        sum_exact += c
 
        sum_steps += carry.steps.float().sum().item()
        sum_phi   += carry.phi.float().sum().item()
        n_samples += B
 
    # DDP reduce — all fields
    totals = ddp_sum_metrics(
        {
            "lm_loss":      sum_lm_loss,
            "halt_penalty": sum_halt_penalty,
            "total_loss":   sum_total_loss,
            "evidence":     sum_evidence,
            "active":       sum_active,
            "n_steps":      float(n_steps_total),
            "accuracy":     sum_acc,
            "exact":        sum_exact,
            "steps_sum":    sum_steps,
            "phi_sum":      sum_phi,
            "n_samples":    float(n_samples),
        },
        device=device,
    )
 
    n_samp  = max(totals["n_samples"], 1.0)
    n_steps = max(totals["n_steps"],   1.0)
    n_act   = max(totals["active"],    1.0)
 
    results = {
        "lm_loss":      totals["lm_loss"]      / n_steps,
        "halt_penalty": totals["halt_penalty"]  / n_steps,
        "total_loss":   totals["total_loss"]    / n_steps,
        "evidence_mean":totals["evidence"]      / n_act,
 
        "accuracy":     totals["accuracy"]      / n_samp,
        "exact_accuracy":totals["exact"]        / n_samp,
        "steps_mean":   totals["steps_sum"]     / n_samp,
        "phi_mean":     totals["phi_sum"]       / n_samp,
 
        "theta": float(model._orig_mod.theta),
    }
 
    return results



@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_cfg: DictConfig):
    cfg = load_synced_config(hydra_cfg)

    rank, world_size, local_rank, device = setup_distributed(cfg)
    is_main = (rank == 0)

    if is_main:
        print(f"[run] project={cfg.project_name} run_name={cfg.run_name}")
        print(f"[dist] rank={rank} world_size={world_size} local_rank={local_rank} device={device}")
    print_gpu_stats(device, rank, world_size)

    train_loader, train_meta = make_dataloader(cfg, split="train", rank=rank, world_size=world_size)
    eval_loader = None
    if os.path.exists(os.path.join(cfg.data_paths[0], "test", "dataset.json")):
        eval_loader, _ = make_dataloader(cfg, split="test", rank=rank, world_size=world_size)

    model = build_model_from_cfg(cfg, train_meta, device)
    if torch.cuda.is_available() and "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model)
        if is_main:
            print("[model] compile enabled")
    loss_head = RTifyLossHead(model)

    psi_params   = [p for n, p in model.named_parameters() if n.endswith(".psi")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith(".psi")]

    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.lr},
            {"params": psi_params, "lr": cfg.lr * 0.1},   # 10× smaller for θ
        ],
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    ckpt_dir = _resolve_ckpt_dir(cfg)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")

    start_epoch = 0
    step = 0
    best_score = None
    wandb_run_id = None

    if cfg.resume and cfg.load_checkpoint:
        if is_main:
            start_epoch, step, best_score, wandb_run_id = load_checkpoint(cfg.load_checkpoint, model, opt, device)
        if dist.is_initialized():
            obj = [start_epoch, step, best_score, wandb_run_id]
            broadcast_objects(obj, src=0)
            start_epoch, step, best_score, wandb_run_id = obj
            broadcast_model(model, src=0)

    use_wandb = is_main and bool(cfg.wandb.get("enabled", False)) and (wandb is not None)
    if use_wandb:
        if cfg.resume and cfg.load_checkpoint and wandb_run_id is not None:
            wandb.init(
                entity=cfg.wandb.get("entity", None),
                project=cfg.wandb.get("project", cfg.project_name),
                id=wandb_run_id,
                resume="must",
                name=cfg.run_name,
                group=cfg.wandb.get("group", "sandbox"),
                mode=cfg.wandb.get("mode", "online"),
                config=cfg.model_dump(),
                tags=build_arch_tags(cfg),
            )
        else:
            wandb.init(
                entity=cfg.wandb.get("entity", None),
                project=cfg.wandb.get("project", cfg.project_name),
                name=cfg.run_name,
                group=cfg.wandb.get("group", "sandbox"),
                mode=cfg.wandb.get("mode", "online"),
                config=cfg.model_dump(),
                tags=build_arch_tags(cfg),
            )

        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        wandb.watch(model, log="parameters", log_freq=100)
        print(f"[W&B] enabled (rank0) run={wandb.run.name if wandb.run else 'None'}")
    elif is_main:
        print("[W&B] disabled")

    steps_per_epoch = sum(1 for _ in iter(train_loader))
    episodes = int(getattr(cfg, "n_reasoning_episodes", 1))
    total_steps = cfg.epochs * steps_per_epoch * max(1, episodes)

    if is_main:
        print(f"[train] steps_per_epoch={steps_per_epoch} episodes={episodes} total_steps={total_steps}")

    master_bar = tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0, leave=True, dynamic_ncols=True, disable=not is_main)
    for epc in master_bar:
        step = train_one_epoch(
            model,
            loss_head,
            train_loader,
            device,
            base_lr=cfg.lr,
            warmup=cfg.lr_warmup_steps,
            total_steps=total_steps,
            min_ratio=cfg.lr_min_ratio,
            step0=step,
            opt=opt,
            use_wandb=use_wandb,
            epoch=epc,
            global_batch_size=cfg.global_batch_size,
            rank=rank,
        )

        do_eval = (eval_loader is not None) and ((epc + 1) % max(1, cfg.eval_interval) == 0)

        if dist.is_initialized():
            flag = torch.tensor([int(do_eval)], device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)   # if any rank is 0 -> all become 0
            do_eval = bool(flag.item())

        if do_eval:
            eval_metrics = evaluate(
                model,
                loss_head,
                eval_loader,
                device,
                max_batches=cfg.max_eval_batches,
                rank=rank,
            )

            if is_main and use_wandb:
                wandb.log({
                    "all.lm_loss": eval_metrics["lm_loss"],
                    "all.halt_penalty": eval_metrics["halt_penalty"],
                    "all.total_loss": eval_metrics["total_loss"],
                    "all.accuracy": eval_metrics["accuracy"],
                    "all.exact_accuracy": eval_metrics["exact_accuracy"],

                    "all.steps_mean": eval_metrics["steps_mean"],
                    "all.phi_mean": eval_metrics["phi_mean"],
                    "all.evidence_mean": eval_metrics["evidence_mean"],
                    "all.theta": eval_metrics["theta"],

                    "epoch": epc + 1,
                    "global_step": step,
                })
                master_bar.set_postfix({
                    "eval_epoch": f"{epc+1}/{cfg.epochs}",
                    "eval_lm_loss": f"{eval_metrics["lm_loss"]:.4f}",
                    "eval_acc": f"{eval_metrics["exact_accuracy"]:.3f}",
                })

            if is_main:
                if cfg.best_metric == "eval_acc_exact":
                    current = eval_metrics["exact_accuracy"]
                elif cfg.best_metric == "eval_acc_all":
                    current = eval_metrics["exact_accuracy"]
                else:
                    current = eval_metrics["lm_loss"]

                better = (best_score is None) or (
                    (cfg.best_mode == "max" and current > best_score) or
                    (cfg.best_mode == "min" and current < best_score)
                )
                if better and cfg.save_best:
                    best_score = current
                    save_checkpoint(best_ckpt, model, opt, epc, step, cfg, best_score=best_score)

        if is_main and ((epc + 1) % max(1, cfg.save_every) == 0):
            save_checkpoint(last_ckpt, model, opt, epc, step, cfg, best_score=best_score)

    if is_main:
        save_checkpoint(last_ckpt, model, opt, cfg.epochs - 1, step, cfg, best_score=best_score)
        if use_wandb:
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()