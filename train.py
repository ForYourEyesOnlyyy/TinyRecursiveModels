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

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from utils.wandb import build_run_name, build_arch_tags
from models.losses import IGNORE_LABEL_ID

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
    """
    Convert Hydra DictConfig into a validated PretrainConfig, and fill in
    reasonable defaults for project_name, run_name, checkpoint_dir, save_every.
    """
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg = TrainConfig(**cfg_dict)

    # project_name
    if cfg.project_name is None:
        proj = cfg.wandb.get("project") if cfg.wandb else None
        if proj is None:
            proj = cfg.arch.name + "-" + os.path.basename(cfg.data_paths[0]).capitalize()
        cfg.project_name = proj

    # run_name

    # for make meaningfull names if not given
    if not hasattr(cfg, "run_name") or cfg.run_name is None:
        cfg.run_name = build_run_name(cfg)

    # checkpoint_dir
    if cfg.checkpoint_dir is None:
        cfg.checkpoint_dir = os.path.join("checkpoints", cfg.project_name, cfg.run_name)

    # save_every: default to eval_interval if not explicitly set
    if cfg.save_every is None:
        cfg.save_every = cfg.eval_interval

    return cfg


def make_dataloader(cfg: TrainConfig, split:str="train"):
    ds_cfg = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths, 
        global_batch_size=cfg.global_batch_size,
        test_set_mode=(split=="test"),
        epochs_per_iter=1,     
        rank=0,
        num_replicas=1
    )
    dataset = PuzzleDataset(ds_cfg, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,   
        num_workers=0, 
        pin_memory=(cfg.device=="cuda")
    )
    return loader, dataset.metadata

def build_model_from_cfg(cfg: TrainConfig, metadata, device: torch.device) -> nn.Module:
    """
    Load model class dynamically
    - cfg.arch.name must be like: "recursive_reasoning.transformers_baseline@SudokuTransformer"
    - metadata supplies runtime fields (seq_len, vocab_size)
    """
    # resolve model class
    model_cls = load_model_class(cfg.arch.name)

    arch_dict = cfg.arch.model_dump()

    for k in ("name", "seq_len", "vocab_size", "num_puzzle_identifiers"):
        arch_dict.pop(k, None)

    # rebuild config with injected runtime params
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



# ---utility---

def _resolve_ckpt_dir(cfg:TrainConfig) -> str:
    # Default: checkpoints/<project>/<run_name>
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
    }
    if torch.cuda.is_available():
        ckpt["rng_torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)
    # print(f"[checkpoint] saved: {path}")

def load_checkpoint(path: str, model, opt, device: torch.device):
    if not (path and os.path.isfile(path)):
        print(f"[checkpoint] no checkpoint at: {path}")
        return 0, 0, None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    # model
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if hasattr(missing, "__len__"):
        if missing or unexpected:
            print(f"[checkpoint] model strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    # optimizer
    if opt is not None and ckpt.get("optimizer_state") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[checkpoint] optimizer state not loaded: {e}")

    # RNG
    try:
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.random.set_rng_state(ckpt["rng_torch"])
        if "rng_torch_cuda" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])
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
    """we wanna make our stuff reproducable"""
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
    """
    compute accuracies of predicted blank points in the puzzle
    we only use it in eval becasue we care only only blanks
    """
    # safely process on one device; should not be a problem, but just in case...
    device = logits.device
    inputs = inputs.to(device)
    labels = labels.to(device)

    preds = logits.argmax(dim=-1)
    blank_mask = (inputs == blank_id)
    valid = (labels != ignore_index) & blank_mask

    # sanity check
    total = valid.float().sum().item()
    if total == 0.:
        return 0.
    predicted = ((preds == labels) & valid).float().sum().item()
    return predicted / total

@torch.no_grad()
def global_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index=0):
    """
    compute accuracies for every token in the sequence
    use for training
    """
    # again, device stuff - just in case...
    device = logits.device
    labels = labels.to(device)

    preds = logits.argmax(dim=-1)
    valid = (labels != ignore_index)

    total = valid.float().sum().item()
    if total == 0.:
        return 0.
    predicted = ((preds == labels) & valid).float().sum().item()
    return predicted / total

# ---training---

def train_one_episode(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    carry,
    opt: torch.optim.Optimizer,
    *,
    blank_id: int,
) -> tuple:
    # Forward one reasoning episode
    carry, final_logits, _ = model(
        inputs,
        carry,
        return_all_logits=False,
    )

    # Compute loss
    loss = F.cross_entropy(
        final_logits.view(-1, final_logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_LABEL_ID,
    )

    # Backward + update
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # Metrics (from final episode only)
    acc_blank = blank_accuracy(final_logits, labels, inputs, blank_id=blank_id)
    acc_all = global_accuracy(final_logits, labels)

    # Detach carry between episodes (TBPTT across episodes)
    carry = type(carry)(
        Z_S=carry.Z_S.detach(),
        Z_R=carry.Z_R.detach(),
    )

    return carry, float(loss.item()), acc_blank, acc_all

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        episodes: int,
        deep_supervision: bool,
        blank_id: int,
        base_lr: float,
        warmup:int,
        total_steps:int, 
        min_ratio: float,
        step0: int,
        opt: torch.optim.Optimizer,
        use_wandb: bool,
        position=1,                                # TQDM
        epoch: int = 0,                            # TQDM
        total_epochs: int = 1,                     # TQDM 
        total_train_batches: int | None = None,    # TQDM
        ) -> int:
    model.train()
    step = step0
    delta_blank_list: List[float] = []
    delta_all_list: List[float] = []
    
    bar = tqdm(
                loader, 
                desc=f"Train ({epoch+1}/{total_epochs})",
                total=total_train_batches,
                position=position, 
                leave=False, 
                dynamic_ncols=True
            )
    for set_name, batch, _ in bar:
        inputs = batch['inputs'].to(device).long()
        labels = batch['labels'].to(device).long()
        carry = model.init_carry(inputs.shape[0], device)

        first_acc_blank = None
        first_acc_all = None
        last_acc_blank = None
        last_acc_all = None
        last_loss_value = None
        # schedule lr
        lr_now = cosine_warmup_lr(step, base_lr, warmup, total_steps, min_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        for ep in range(episodes):
            step += 1
            carry, loss, acc_blank, acc_all = train_one_episode(
            model,
            inputs,
            labels,
            carry,
            opt,
            blank_id=blank_id,
            )

            if ep == 0:
                first_acc_blank = acc_blank
                first_acc_all = acc_all

            if ep == episodes - 1:
                last_acc_blank = acc_blank
                last_acc_all = acc_all
                last_loss_value = loss


        delta_blank = 0.0
        delta_all = 0.0
        if first_acc_blank is not None and last_acc_blank is not None:
            delta_blank = float(last_acc_blank - first_acc_blank)
            delta_all = float(last_acc_all - first_acc_all)
            delta_blank_list.append(delta_blank)
            delta_all_list.append(delta_all)

        bar.set_postfix({
            "loss": f"{(last_loss_value or 0.0):.4f}",
            "blank_acc": f"{(last_acc_blank or 0.0):.3f}",
            "all_acc": f"{(last_acc_all or 0.0):.3f}",
            "Δblank": f"{delta_blank:+.3f}",
        })


        if use_wandb and wandb is not None:
            wandb.log({
                "train/loss_ce": last_loss_value,
                "train/acc_blank": last_acc_blank,
                "train/acc_all": last_acc_all,
                # "train/delta_blank_acc": delta_blank,
                # "train/delta_all_acc": delta_all,
                # "train/episodes": episodes,
                "step": step,
                "epoch": epoch + 1,
            })
    bar.close()

    if use_wandb and wandb is not None:
        avg_delta_blank = sum(delta_blank_list) / max(1, len(delta_blank_list))
        avg_delta_all = sum(delta_all_list) / max(1, len(delta_all_list))

        wandb.log({
            "reasoning_effectiveness/train_delta_acc": avg_delta_blank,
            # "reasoning_effectiveness/avg_delta_all_acc": avg_delta_all,
            # "reasoning_effectiveness/episodes": episodes,
            "epoch": epoch + 1,
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
    max_batches: int | None = None,
    position: int = 2,
    epoch: int = 0,
    total_epochs: int = 1,
) -> tuple[float, float, float, float, float]:
    model.eval()

    losses = []
    acc_all_final = []
    acc_blank_final = []
    delta_blank_list = []
    delta_all_list = []

    iterable = loader if max_batches is None else islice(loader, max_batches)
    bar = tqdm(
                iterable, 
                desc=f"Eval ({epoch+1}/{total_epochs})", 
                total=max_batches, 
                position=position, 
                leave=False, 
                dynamic_ncols=True
            )
    for set_name, batch, _ in bar:
        inputs = batch['inputs'].to(device).long()
        labels = batch['labels'].to(device).long()

        carry = model.init_carry(inputs.shape[0], device)

        first_acc_blank = None
        first_acc_all = None
        last_logits = None

        for ep in range(max(1, episodes_eval)):
            carry, logits, _ = model.forward(inputs, carry, return_all_logits=False)
            last_logits = logits

            acc_b = blank_accuracy(logits, labels, inputs, blank_id=blank_id)
            acc_a = global_accuracy(logits, labels)

            if ep == 0:
                first_acc_blank = acc_b
                first_acc_all = acc_a

        # final loss/acc on last episode
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

        # delta: last - first
        if first_acc_blank is not None:
            delta_blank_list.append(float(final_acc_b - first_acc_blank))
            delta_all_list.append(float(final_acc_a - first_acc_all))

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


# ---hyra---

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_cfg: DictConfig):
    # Convert & validate once
    cfg = load_synced_config(hydra_cfg)

    set_seed(cfg.seed)
    device = get_device(cfg)
    print(f"[device] {device}")

    train_loader, train_meta = make_dataloader(cfg, split='train')
    eval_loader, eval_meta = (make_dataloader(cfg, split='test') if os.path.exists(
        os.path.join(cfg.data_paths[0], "test", "dataset.json")) else (None, train_meta))
    
    blank_id = cfg.blank_id
    model = build_model_from_cfg(cfg, train_meta, device) # model is on device
    deep_supervision = getattr(cfg.arch, "deep_supervision", False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2)
    )

    use_wandb = bool(getattr(cfg, "wandb", {}).get("enabled", False)) and (wandb is not None)
    if use_wandb and not (cfg.load_checkpoint and cfg.resume):
        wandb.init(
            entity="mrtshv-innopolis-university",
            project=cfg.wandb.get("project", "baseline"),
            name=getattr(cfg, "run_name", "run"),
            group = cfg.wandb.get("group", "sandbox"),
            mode=cfg.wandb.get("mode", "online"),
            config=cfg.model_dump(),
            tags=build_arch_tags(cfg),
        )
        wandb.watch(model)
        print(f"[W&B] Logging enabled — run: {wandb.run.name if wandb.run else 'None'}")
    else:
        print("[W&B] Logging disabled.")

    steps_per_epoch = sum(1 for _ in iter(train_loader))
    total_steps = cfg.epochs * steps_per_epoch

    train_loader, _ = make_dataloader(cfg, split="train")


    # --- Checkpoint resume ---
    ckpt_dir = _resolve_ckpt_dir(cfg)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    best_score = None

    start_epoch = 0
    step = 0

    if cfg.resume and cfg.load_checkpoint:
        start_epoch, step, best_score, wandb_run_id = load_checkpoint(cfg.load_checkpoint, model, opt, device)
        if wandb_run_id is None:
            wandb_run_id =cfg.wandb.get("resume_run_id", None)
        wandb.init(
            entity="mrtshv-innopolis-university",
            project=cfg.wandb.get("project", "baseline"),
            id=wandb_run_id,
            resume="must",
            name=cfg.run_name,
            group=cfg.wandb.get("group", "sandbox"),
            mode=cfg.wandb.get("mode", "online"),
            config=cfg.model_dump(),
            tags=build_arch_tags(cfg),
        )
        wandb.watch(model)
        print(f"[W&B] Logging enabled — run: {wandb.run.name if wandb.run else 'None'}")

    master_bar = tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0, leave=True, dynamic_ncols=True)
    for epoch in master_bar:
        step = train_one_epoch(
            model, train_loader, device,
            episodes=cfg.n_reasoning_episodes,
            deep_supervision=deep_supervision,
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
            total_train_batches=steps_per_epoch
        )
        # eval periodically to speedup runtime
        if (epoch + 1) % max(1, cfg.eval_interval) == 0 and eval_loader is not None:
            eval_loss, eval_acc_all, eval_acc_blank, eval_reasoning_effect = evaluate(
                model, eval_loader, device, 
                episodes_eval=cfg.n_reasoning_episodes,
                # deep_supervision=deep_supervision, 
                blank_id=blank_id, 
                max_batches=cfg.max_eval_batches, 
                position=2,
                epoch=epoch,
                total_epochs=cfg.epochs,
            )
            if use_wandb:
                wandb.log({
                    "eval/loss_ce": eval_loss,
                    "eval/acc_all": eval_acc_all,
                    "eval/acc_blank": eval_acc_blank,
                    "reasoning_effectiveness/eval_delta_acc": eval_reasoning_effect,
                    "epoch": epoch + 1
                })
            master_bar.set_postfix({
            "eval_epoch": f"{epoch+1}/{cfg.epochs}",
            "eval_loss": f"{eval_loss:.4f}",
            "eval_blank_acc": f"{eval_acc_blank:.3f}",
            "eval_episode_delta": f"{eval_reasoning_effect:.3f}"
            })

            # save best
            if getattr(cfg, "save_best", True):
                metric_name = getattr(cfg, "best_metric", "eval_acc_blank")
                mode = getattr(cfg, "best_mode", "max")
                current = eval_acc_blank if metric_name == "eval_acc_blank" else eval_loss
                better = (best_score is None) or ((mode == "max" and current > best_score) or (mode == "min" and current < best_score))
                if better:
                    best_score = current
                    save_checkpoint(best_ckpt, model, opt, epoch, step, cfg, best_score=best_score)

         # save last every N epochs 
        if (epoch + 1) % max(1, getattr(cfg, "save_every", cfg.eval_interval)) == 0:
            save_checkpoint(last_ckpt, model, opt, epoch, step, cfg, best_score=best_score)
    
    # final save
    save_checkpoint(last_ckpt, model, opt, cfg.epochs - 1, step, cfg, best_score=best_score)

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
    




