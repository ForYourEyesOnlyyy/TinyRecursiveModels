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
from models.losses import stablemax_cross_entropy

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------
# Config
# -------------------------

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    arch: ArchConfig

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None

    # Data
    data_paths: List[str]
    data_split_train: str
    data_split_eval: str
    blank_id: int
    global_batch_size: int
    max_eval_batches: Optional[int] = None

    # Optimizer
    lr: float
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999

    # Training
    seed: int = 0
    epochs: int
    eval_interval: int
    device: str = "auto"  # "cpu" | "cuda" | "mps" | "auto"

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_every: Optional[int] = None
    save_best: bool = True
    resume: bool = False
    load_checkpoint: Optional[str] = None
    best_metric: str = "eval_acc_blank"
    best_mode: str = "max"  # "max" or "min"

    # Logging
    wandb: dict[str, Any] = {}


def load_synced_config(hydra_cfg: DictConfig) -> TrainConfig:
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg = TrainConfig(**cfg_dict)

    # project_name default
    if cfg.project_name is None:
        proj = cfg.wandb.get("project") if cfg.wandb else None
        if proj is None:
            proj = cfg.arch.name + "-" + os.path.basename(cfg.data_paths[0]).capitalize()
        cfg.project_name = proj

    # run_name default
    if cfg.run_name is None:
        cfg.run_name = build_run_name(cfg)

    # checkpoint dir default
    if cfg.checkpoint_dir is None:
        cfg.checkpoint_dir = os.path.join("checkpoints", cfg.project_name, cfg.run_name)

    # save_every default
    if cfg.save_every is None:
        cfg.save_every = cfg.eval_interval

    return cfg


# -------------------------
# Distributed helpers
# -------------------------

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

    # Device per rank
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
    """Average gradients across ranks (manual DDP)."""
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


# -------------------------
# Data / model
# -------------------------

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


# -------------------------
# Checkpointing
# -------------------------

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

    # Load on CPU to avoid RNG device issues
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


# -------------------------
# Misc utils
# -------------------------

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
def blank_accuracy(logits: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor, *, blank_id=1, ignore_index=0) -> float:
    preds = logits.argmax(dim=-1)
    blank_mask = (inputs == blank_id)
    valid = (labels != ignore_index) & blank_mask
    total = valid.float().sum().item()
    if total == 0.0:
        return 0.0
    correct = ((preds == labels) & valid).float().sum().item()
    return correct / total


@torch.no_grad()
def global_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index=0) -> float:
    preds = logits.argmax(dim=-1)
    valid = (labels != ignore_index)
    total = valid.float().sum().item()
    if total == 0.0:
        return 0.0
    correct = ((preds == labels) & valid).float().sum().item()
    return correct / total


# -------------------------
# Training / Eval
# -------------------------

def train_one_episode(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    carry,
    opt: torch.optim.Optimizer,
    *,
    blank_id: int,
) -> tuple:
    # One reasoning episode (your TRM forward signature)
    carry, logits = model(inputs, carry)

    # Stablemax CE (per-token), then reduce with ignore mask
    mask = (labels != IGNORE_LABEL_ID)                 # [B, L] bool
    loss_counts = mask.sum(-1)                         # [B]
    loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # [B, 1]

    per_token = stablemax_cross_entropy(
        logits,
        labels,
        ignore_index=IGNORE_LABEL_ID,
        valid_mask=mask,
    )                                                  # [B, L]

    loss = (per_token / loss_divisor).sum()            # scalar (exact TRM)

    opt.zero_grad(set_to_none=True)
    loss.backward()

    # distributed gradient sync
    allreduce_grads(model)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    acc_blank = blank_accuracy(logits, labels, inputs, blank_id=blank_id)
    acc_all = global_accuracy(logits, labels)

    return carry, float(loss.item()), acc_blank, acc_all


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    episodes: int,
    blank_id: int,
    base_lr: float,
    warmup: int,
    total_steps: int,
    min_ratio: float,
    step0: int,
    opt: torch.optim.Optimizer,
    use_wandb: bool,
    epoch: int,
    total_epochs: int,
    total_train_batches: Optional[int],
    rank: int,
) -> int:
    model.train()
    step = step0

    for _, batch, _ in loader:
        inputs = batch["inputs"].to(device).long()
        labels = batch["labels"].to(device).long()

        carry = model.init_carry(inputs.shape[0], device)

        last_loss_val = None
        last_acc_blank = None
        last_acc_all = None

        for _ep in range(max(1, episodes)):
            # LR per optimizer step
            lr_now = cosine_warmup_lr(step, base_lr, warmup, total_steps, min_ratio)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            carry, loss_val, acc_blank, acc_all = train_one_episode(
                model, inputs, labels, carry, opt, blank_id=blank_id
            )

            last_loss_val = loss_val
            last_acc_blank = acc_blank
            last_acc_all = acc_all

            step += 1

        # Log once per batch (rank 0 only), same behavior as before
        if rank == 0 and use_wandb and wandb is not None:
            wandb.log({
                "train/loss_ce": last_loss_val,
                "train/acc_blank": last_acc_blank,
                "train/acc_all": last_acc_all,
                "epoch": epoch + 1,
                "global_step": step,
            })

    return step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    episodes_eval: int,
    blank_id: int,
    max_batches: Optional[int],
    epoch: int,
    total_epochs: int,
    rank: int,
) -> Tuple[float, float, float, float]:
    model.eval()

    losses: List[float] = []
    acc_all: List[float] = []
    acc_blank: List[float] = []

    iterable = loader if max_batches is None else islice(loader, max_batches)
    bar = tqdm(
        iterable,
        desc=f"Eval ({epoch+1}/{total_epochs})",
        total=max_batches,
        position=1,
        leave=False,
        dynamic_ncols=True,
        disable=(rank != 0),
    )

    for _, batch, _ in bar:
        inputs = batch["inputs"].to(device).long()
        labels = batch["labels"].to(device).long()

        carry = model.init_carry(inputs.shape[0], device)

        for ep in range(max(1, episodes_eval)):
            carry, logits = model(inputs, carry)

        mask = (labels != IGNORE_LABEL_ID)              # [B, L]
        loss_counts = mask.sum(-1)                      # [B]
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        per_token = stablemax_cross_entropy(
            logits,
            labels,
            ignore_index=IGNORE_LABEL_ID,
            valid_mask=mask,
        )                                               # [B, L]

        loss = (per_token / loss_divisor).sum().item() # scalar

        acc_b = blank_accuracy(logits, labels, inputs, blank_id=blank_id)
        acc_a = global_accuracy(logits, labels)

        losses.append(loss)
        acc_blankl.append(acc_b)
        acc_all.append(acc_a)

        if rank == 0:
            bar.set_postfix({
                "loss": f"{loss:.4f}",
                "blank_acc": f"{acc_b:.3f}",
            })

    bar.close()

    n = max(1, len(losses))
    avg_loss = sum(losses) / n
    avg_acc_all = sum(acc_all) / n
    avg_acc_blank = sum(acc_blank) / n

    return avg_loss, avg_acc_all, avg_acc_blank


# -------------------------
# Main
# -------------------------

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_cfg: DictConfig):
    cfg = load_synced_config(hydra_cfg)

    rank, world_size, local_rank, device = setup_distributed(cfg)
    is_main = (rank == 0)

    # Useful startup prints (only once per process)
    if is_main:
        print(f"[run] project={cfg.project_name} run_name={cfg.run_name}")
        print(f"[dist] rank={rank} world_size={world_size} local_rank={local_rank} device={device}")
    print_gpu_stats(device, rank, world_size)

    # Dataloaders
    train_loader, train_meta = make_dataloader(cfg, split="train", rank=rank, world_size=world_size)
    eval_loader = None
    if os.path.exists(os.path.join(cfg.data_paths[0], "test", "dataset.json")):
        eval_loader, _ = make_dataloader(cfg, split="test", rank=rank, world_size=world_size)

    # Model + optimizer
    model = build_model_from_cfg(cfg, train_meta, device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    # Resume (rank 0 loads, then broadcast)
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

    # W&B (rank 0 only)
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
        wandb.watch(model)
        print(f"[W&B] enabled (rank0) run={wandb.run.name if wandb.run else 'None'}")
    elif is_main:
        print("[W&B] disabled")

    # Steps per epoch + total_steps
    steps_per_epoch = sum(1 for _ in iter(train_loader))
    episodes = int(getattr(cfg, "n_reasoning_episodes", 1))
    total_steps = cfg.epochs * steps_per_epoch * max(1, episodes)

    if is_main:
        print(f"[train] steps_per_epoch={steps_per_epoch} episodes={episodes} total_steps={total_steps}")

    # Training loop
    master_bar = tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0, leave=True, dynamic_ncols=True, disable=not is_main)
    for epc in master_bar:
        step = train_one_epoch(
            model,
            train_loader,
            device,
            episodes=episodes,
            blank_id=cfg.blank_id,
            base_lr=cfg.lr,
            warmup=cfg.lr_warmup_steps,
            total_steps=total_steps,
            min_ratio=cfg.lr_min_ratio,
            step0=step,
            opt=opt,
            use_wandb=use_wandb,
            epoch=epc,
            total_epochs=cfg.epochs,
            total_train_batches=steps_per_epoch,
            rank=rank,
        )

        # Eval only on rank 0 (simple + correct)
        if is_main and eval_loader is not None and ((epc + 1) % max(1, cfg.eval_interval) == 0):
            eval_loss, eval_acc_all, eval_acc_blank, eval_delta = evaluate(
                model,
                eval_loader,
                device,
                episodes_eval=episodes,
                blank_id=cfg.blank_id,
                max_batches=cfg.max_eval_batches,
                epoch=epc,
                total_epochs=cfg.epochs,
                rank=rank,
            )
            if use_wandb:
                wandb.log({
                    "eval/loss_ce": eval_loss,
                    "eval/acc_all": eval_acc_all,
                    "eval/acc_blank": eval_acc_blank,
                    "epoch": epc + 1,
                    "global_step": step,
                })
                master_bar.set_postfix({
                    "eval_epoch": f"{epoch+1}/{cfg.epochs}",
                    "eval_loss": f"{eval_loss:.4f}",
                    "eval_blank_acc": f"{eval_acc_blank:.3f}"
                })

            # Save best on rank 0
            current = eval_acc_blank if cfg.best_metric == "eval_acc_blank" else eval_loss
            better = (best_score is None) or ((cfg.best_mode == "max" and current > best_score) or (cfg.best_mode == "min" and current < best_score))
            if better and cfg.save_best:
                best_score = current
                save_checkpoint(best_ckpt, model, opt, epc, step, cfg, best_score=best_score)

        # Save last checkpoint on rank 0
        if is_main and ((epc + 1) % max(1, cfg.save_every) == 0):
            save_checkpoint(last_ckpt, model, opt, epc, step, cfg, best_score=best_score)

    # Final save + shutdown
    if is_main:
        save_checkpoint(last_ckpt, model, opt, cfg.epochs - 1, step, cfg, best_score=best_score)
        if use_wandb:
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()