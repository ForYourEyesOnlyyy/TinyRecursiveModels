#   torchrun --standalone --nproc_per_node=4 train.py device=cuda global_batch_size=512 ...

import os
import math
import random
from itertools import islice
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import pydantic
from tqdm import tqdm
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from utils.wandb import build_run_name, build_arch_tags
from models.losses import IGNORE_LABEL_ID


# ------------------------------ Configs ------------------------------

class ArchConfig(pydantic.BaseModel):
    """
    Architecture config: holds model class name and arbitrary extra hyperparams
    from `arch:` in the YAML.
    """
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class TrainConfig(pydantic.BaseModel):
    """
    Global training config (TRM-style), constructed once from Hydra's DictConfig.
    """
    model_config = pydantic.ConfigDict(extra='allow')

    # Architecture sub-config
    arch: ArchConfig

    # Names (explicit fields to avoid extra-field ambiguity)
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

    # Reasoning episodes (multi-episode TRM)
    n_reasoning_episodes: int = 1

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
    """
    Convert Hydra DictConfig into a validated TrainConfig and fill defaults:
      - project_name
      - run_name
      - checkpoint_dir
      - save_every
    """
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


# ------------------------------ DDP helpers ------------------------------

def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ


def get_rank_world() -> tuple[int, int, int]:
    """
    Returns: (rank, world_size, local_rank)
    """
    if not is_distributed():
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size, local_rank


def print_gpu_stats(rank: int, world_size: int, local_rank: int, cfg: TrainConfig):
    """
    Print GPU availability and devices used. Only rank 0 prints.
    """
    if rank != 0:
        return

    cuda_avail = torch.cuda.is_available()
    n_cuda = torch.cuda.device_count() if cuda_avail else 0

    print("\n[GPU] CUDA available:", cuda_avail)
    print("[GPU] CUDA device count:", n_cuda)

    if cuda_avail:
        for i in range(n_cuda):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024 ** 3)
            print(f"[GPU]  cuda:{i}  {name}  {mem_gb:.1f} GB")

    if cfg.device == "cuda" and cuda_avail:
        print(f"[GPU] Requested device=cuda, using {world_size} process(es) / GPU(s)")
    else:
        print(f"[GPU] Requested device={cfg.device}, DDP enabled: {is_distributed()}")

    if is_distributed():
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}\n")
    else:
        print("[DDP] disabled\n")


# ------------------------------ Data / model ------------------------------

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
        num_workers=0,
        pin_memory=(cfg.device == "cuda"),
    )
    return loader, dataset.metadata


def build_model_from_cfg(cfg: TrainConfig, metadata, device: torch.device) -> nn.Module:
    """
    Instantiate model from cfg.arch, injecting runtime seq_len and vocab_size.
    """
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
    print(f"[model] Loaded model: {cfg.arch.name}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable params: {trainable_params}")
    return model


# ------------------------------ Checkpointing ------------------------------

def _resolve_ckpt_dir(cfg: TrainConfig) -> str:
    project = (cfg.wandb.get("project", None) if getattr(cfg, "wandb", None) else None) \
              or getattr(cfg, "project_name", "baseline")
    run_name = getattr(cfg, "run_name", "run")
    base = cfg.checkpoint_dir or os.path.join("checkpoints", project, run_name)
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

        # RNG states for reproducibility
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,

        # W&B run id (for true resume)
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, opt, device: torch.device):
    if not (path and os.path.isfile(path)):
        print(f"[checkpoint] no checkpoint at: {path}")
        return 0, 0, None, None

    # Load on CPU for RNG safety (PyTorch 2.6+ safe behavior)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[checkpoint] model strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    if opt is not None and ckpt.get("optimizer_state") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[checkpoint] optimizer state not loaded: {e}")

    # RNG restore (best-effort)
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

        if torch.cuda.is_available() and ckpt.get("rng_torch_cuda") is not None:
            cuda_states = ckpt["rng_torch_cuda"]
            fixed = []
            for s in cuda_states:
                if isinstance(s, np.ndarray):
                    s = torch.from_numpy(s)
                s = s.to("cpu")
                if s.dtype != torch.uint8:
                    s = s.to(torch.uint8)
                fixed.append(s)
            torch.cuda.set_rng_state_all(fixed)
    except Exception as e:
        print(f"[checkpoint] RNG restore failed: {e}")

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    best_score = ckpt.get("best_score", None)
    wandb_run_id = ckpt.get("wandb_run_id", None)
    print(f"[checkpoint] loaded: {path} (epoch={epoch}, step={step}, best={best_score}, id={wandb_run_id})")
    return epoch, step, best_score, wandb_run_id


# ------------------------------ Misc utils ------------------------------

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


@torch.no_grad()
def blank_accuracy(logits: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor, *, blank_id=1, ignore_index=0) -> float:
    device = logits.device
    inputs = inputs.to(device)
    labels = labels.to(device)

    preds = logits.argmax(dim=-1)
    blank_mask = (inputs == blank_id)
    valid = (labels != ignore_index) & blank_mask

    total = valid.float().sum().item()
    if total == 0.0:
        return 0.0
    predicted = ((preds == labels) & valid).float().sum().item()
    return predicted / total


@torch.no_grad()
def global_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index=0):
    device = logits.device
    labels = labels.to(device)

    preds = logits.argmax(dim=-1)
    valid = (labels != ignore_index)

    total = valid.float().sum().item()
    if total == 0.0:
        return 0.0
    predicted = ((preds == labels) & valid).float().sum().item()
    return predicted / total


# ------------------------------ Training (multi-episode) ------------------------------

def train_one_episode(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    carry,
    opt: torch.optim.Optimizer,
    *,
    blank_id: int,
) -> tuple:
    """
    One reasoning episode:
      - forward(inputs, carry) -> (carry, logits, _)
      - CE loss
      - optimizer step
      - detach carry
    """
    carry, final_logits, _ = model(inputs, carry, return_all_logits=False)

    loss = F.cross_entropy(
        final_logits.view(-1, final_logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_LABEL_ID,
    )

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    acc_blank = blank_accuracy(final_logits, labels, inputs, blank_id=blank_id)
    acc_all = global_accuracy(final_logits, labels)

    # Detach carry between episodes (TBPTT across episodes)
    carry = type(carry)(Z_S=carry.Z_S.detach(), Z_R=carry.Z_R.detach())
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
    position=1,
    epoch: int = 0,
    total_epochs: int = 1,
    total_train_batches: int | None = None,
) -> int:
    model.train()
    step = step0

    delta_blank_list: List[float] = []

    bar = tqdm(
        loader,
        desc=f"Train ({epoch+1}/{total_epochs})",
        total=total_train_batches,
        position=position,
        leave=False,
        dynamic_ncols=True,
    )

    for set_name, batch, _ in bar:
        inputs = batch["inputs"].to(device).long()
        labels = batch["labels"].to(device).long()

        carry = model.init_carry(inputs.shape[0], device)

        first_acc_blank = None
        last_acc_blank = None
        last_acc_all = None
        last_loss_value = None

        for ep in range(max(1, episodes)):
            # LR schedule per optimizer update (episode step)
            lr_now = cosine_warmup_lr(step, base_lr, warmup, total_steps, min_ratio)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            carry, loss_value, acc_blank, acc_all = train_one_episode(
                model,
                inputs,
                labels,
                carry,
                opt,
                blank_id=blank_id,
            )

            if ep == 0:
                first_acc_blank = acc_blank

            if ep == episodes - 1:
                last_acc_blank = acc_blank
                last_acc_all = acc_all
                last_loss_value = loss_value

            step += 1  # optimizer-step counter

        delta_blank = 0.0
        if first_acc_blank is not None and last_acc_blank is not None:
            delta_blank = float(last_acc_blank - first_acc_blank)
            delta_blank_list.append(delta_blank)

        bar.set_postfix({
            "loss": f"{(last_loss_value or 0.0):.4f}",
            "blank_acc": f"{(last_acc_blank or 0.0):.3f}",
            "all_acc": f"{(last_acc_all or 0.0):.3f}",
            "Δblank": f"{delta_blank:+.3f}",
        })

        if use_wandb and wandb is not None:
            wandb.log(
                {
                    "train/loss_ce": last_loss_value,
                    "train/acc_blank": last_acc_blank,
                    "train/acc_all": last_acc_all,
                    "train/delta_blank_acc": delta_blank,
                    "global_step": step,
                    "epoch": epoch + 1,
                }
            )

    bar.close()

    if use_wandb and wandb is not None:
        avg_delta_blank = sum(delta_blank_list) / max(1, len(delta_blank_list))
        wandb.log(
            {
                "reasoning_effectiveness/avg_delta_blank_acc": avg_delta_blank,
                "epoch": epoch + 1,
                "global_step": step,
            }
        )

    return step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    episodes_eval: int,
    blank_id: int,
    max_batches: int | None = None,
    position: int = 2,
    epoch: int = 0,
    total_epochs: int = 1,
) -> tuple[float, float, float, float]:
    model.eval()

    losses: List[float] = []
    acc_all_final: List[float] = []
    acc_blank_final: List[float] = []
    delta_blank_list: List[float] = []

    iterable = loader if max_batches is None else islice(loader, max_batches)
    bar = tqdm(
        iterable,
        desc=f"Eval ({epoch+1}/{total_epochs})",
        total=max_batches,
        position=position,
        leave=False,
        dynamic_ncols=True,
    )

    for set_name, batch, _ in bar:
        inputs = batch["inputs"].to(device).long()
        labels = batch["labels"].to(device).long()

        carry = model.init_carry(inputs.shape[0], device)

        first_acc_blank = None
        last_logits = None

        for ep in range(max(1, episodes_eval)):
            carry, logits, _ = model(inputs, carry, return_all_logits=False)
            last_logits = logits

            acc_b = blank_accuracy(logits, labels, inputs, blank_id=blank_id)
            if ep == 0:
                first_acc_blank = acc_b

        loss = F.cross_entropy(
            last_logits.view(-1, last_logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
        ).item()

        final_acc_b = blank_accuracy(last_logits, labels, inputs, blank_id=blank_id)
        final_acc_a = global_accuracy(last_logits, labels)

        losses.append(loss)
        acc_blank_final.append(final_acc_b)
        acc_all_final.append(final_acc_a)

        if first_acc_blank is not None:
            delta_blank_list.append(float(final_acc_b - first_acc_blank))

        bar.set_postfix({
            "loss": f"{loss:.4f}",
            "blank_acc": f"{final_acc_b:.3f}",
            "Δblank": f"{(delta_blank_list[-1] if delta_blank_list else 0.0):+.3f}",
        })

    bar.close()

    n = max(1, len(losses))
    avg_loss = sum(losses) / n
    avg_acc_all = sum(acc_all_final) / n
    avg_acc_blank = sum(acc_blank_final) / n
    avg_delta_blank = sum(delta_blank_list) / max(1, len(delta_blank_list))

    return avg_loss, avg_acc_all, avg_acc_blank, avg_delta_blank


# ------------------------------ Main ------------------------------

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_cfg: DictConfig):
    cfg = load_synced_config(hydra_cfg)

    rank, world_size, local_rank = get_rank_world()

    # Init DDP (CUDA only)
    if is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Device per rank
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed() else "cuda")
    else:
        device = get_device(cfg)

    # Seeds per rank
    set_seed(cfg.seed + rank)

    print_gpu_stats(rank, world_size, local_rank, cfg)

    # Data
    train_loader, train_meta = make_dataloader(cfg, split="train", rank=rank, world_size=world_size)
    has_eval = os.path.exists(os.path.join(cfg.data_paths[0], "test", "dataset.json"))
    eval_loader, eval_meta = (make_dataloader(cfg, split="test", rank=rank, world_size=world_size) if has_eval else (None, train_meta))

    blank_id = cfg.blank_id

    # Model
    model = build_model_from_cfg(cfg, train_meta, device)

    # Wrap with DDP
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    core = model.module if isinstance(model, DDP) else model
    deep_supervision = getattr(cfg.arch, "deep_supervision", False)

    # Optimizer (DDP wrapper params)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    # Rank-0 only logging / checkpointing
    is_main = (rank == 0)
    use_wandb = is_main and bool(cfg.wandb.get("enabled", False)) and (wandb is not None)

    # Steps per epoch (each rank has its own iterator; this is approximate but fine)
    steps_per_epoch = sum(1 for _ in iter(train_loader))

    # Total optimizer steps: epochs * batches_per_epoch * episodes
    total_steps = cfg.epochs * steps_per_epoch * max(1, cfg.n_reasoning_episodes)

    # Checkpoints
    ckpt_dir = _resolve_ckpt_dir(cfg)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    best_score = None

    start_epoch = 0
    step = 0
    wandb_run_id = None

    # Resume checkpoint (all ranks load so weights match)
    if cfg.resume and cfg.load_checkpoint:
        start_epoch, step, best_score, wandb_run_id = load_checkpoint(cfg.load_checkpoint, model, opt, device)

    # W&B init (rank 0 only)
    if use_wandb:
        if cfg.resume and cfg.load_checkpoint:
            if wandb_run_id is None:
                wandb_run_id = cfg.wandb.get("resume_run_id", None)
            wandb.init(
                entity=cfg.wandb.get("entity", None),
                project=cfg.wandb.get("project", "baseline"),
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
                project=cfg.wandb.get("project", "baseline"),
                name=cfg.run_name,
                group=cfg.wandb.get("group", "sandbox"),
                mode=cfg.wandb.get("mode", "online"),
                config=cfg.model_dump(),
                tags=build_arch_tags(cfg),
            )

        # Use global_step as x-axis everywhere
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

        wandb.watch(core)
        print(f"[W&B] Logging enabled — run: {wandb.run.name if wandb.run else 'None'}")
    elif is_main:
        print("[W&B] Logging disabled.")

    # Training loop
    master_iter = tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0, leave=True, dynamic_ncols=True) if is_main else range(start_epoch, cfg.epochs)

    for epoch in master_iter:
        # Train on all ranks
        step = train_one_epoch(
            core,
            train_loader,
            device,
            episodes=cfg.n_reasoning_episodes,
            blank_id=blank_id,
            base_lr=cfg.lr,
            warmup=cfg.lr_warmup_steps,
            total_steps=total_steps,
            min_ratio=cfg.lr_min_ratio,
            step0=step,
            opt=opt,
            use_wandb=use_wandb,
            position=1,
            epoch=epoch,
            total_epochs=cfg.epochs,
            total_train_batches=steps_per_epoch,
        )

        # Eval only on rank 0 for simplicity
        if is_main and eval_loader is not None and ((epoch + 1) % max(1, cfg.eval_interval) == 0):
            eval_loss, eval_acc_all, eval_acc_blank, eval_delta = evaluate(
                core,
                eval_loader,
                device,
                episodes_eval=cfg.n_reasoning_episodes,
                blank_id=blank_id,
                max_batches=cfg.max_eval_batches,
                position=2,
                epoch=epoch,
                total_epochs=cfg.epochs,
            )

            if use_wandb:
                wandb.log(
                    {
                        "eval/loss_ce": eval_loss,
                        "eval/acc_all": eval_acc_all,
                        "eval/acc_blank": eval_acc_blank,
                        "reasoning_effectiveness/eval_delta_blank_acc": eval_delta,
                        "epoch": epoch + 1,
                        "global_step": step,
                    }
                )

            if isinstance(master_iter, tqdm):
                master_iter.set_postfix({
                    "eval_loss": f"{eval_loss:.4f}",
                    "eval_blank_acc": f"{eval_acc_blank:.3f}",
                    "eval_Δblank": f"{eval_delta:+.3f}",
                })

            # Save best
            if cfg.save_best:
                metric_name = cfg.best_metric
                mode = cfg.best_mode
                current = eval_acc_blank if metric_name == "eval_acc_blank" else eval_loss
                better = (best_score is None) or ((mode == "max" and current > best_score) or (mode == "min" and current < best_score))
                if better:
                    best_score = current
                    save_checkpoint(best_ckpt, model, opt, epoch, step, cfg, best_score=best_score)

        # Save last periodically (rank 0)
        if is_main and ((epoch + 1) % max(1, (cfg.save_every or cfg.eval_interval)) == 0):
            save_checkpoint(last_ckpt, model, opt, epoch, step, cfg, best_score=best_score)

    # Final save
    if is_main:
        save_checkpoint(last_ckpt, model, opt, cfg.epochs - 1, step, cfg, best_score=best_score)

    if use_wandb:
        wandb.finish()

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()