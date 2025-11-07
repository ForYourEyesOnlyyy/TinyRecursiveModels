import os, math, json
from itertools import islice
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import importlib
import pydantic
import hydra
from functools import partial

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from models.recursive_reasoning.transformers_baseline import SudokuTransformerConfig
from utils.functions import load_model_class
from models.losses import IGNORE_LABEL_ID
import random
from tqdm import tqdm
import wandb

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

def make_dataloader(cfg, split="train"):
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

def build_model_from_cfg(cfg: DictConfig, metadata, device: torch.device) -> nn.Module:
    """
    Load model class dynamically
    - cfg.arch.name must be like: "recursive_reasoning.transformers_baseline@SudokuTransformer"
    - metadata supplies runtime fields (seq_len, vocab_size)
    """
    # resolve model class
    model_cls = load_model_class(cfg.arch.name)

    arch_dict = OmegaConf.to_container(cfg.arch, resolve=True)

    # rebuild config with injected runtime params
    model_cfg = {
        **arch_dict,
        "seq_len": int(metadata.seq_len),
        "vocab_size": int(metadata.vocab_size)
    }

    model_cfg = SudokuTransformerConfig(**model_cfg)
    model = model_cls(model_cfg).to(device)
    return model



# ---utility---
def get_device(cfg: DictConfig) -> torch.device:
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


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        blank_id: int,
        base_lr: float,
        warmup:int,
        total_steps:int, 
        min_ratio: float,
        step0: int,
        opt: torch.optim.Optimizer,
        use_wandb: bool,
        position=1  # TQDM pretty stuff
        ) -> int:
    model.train()
    step = step0
    
    bar = tqdm(loader, desc=f"Train",
               position=position, leave=False)
    for set_name, batch, _ in bar:
        step += 1
        inputs = batch['inputs'].to(device)
        labels = batch['labels'].to(device)

        logits = model(inputs)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_LABEL_ID
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # schedule lr
        lr_now = cosine_warmup_lr(step, base_lr, warmup, total_steps, min_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        # metrics and logging
        acc_blank = blank_accuracy(logits, labels, inputs, blank_id=blank_id)
        acc_all   = global_accuracy(logits, labels)

        bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "blank_acc": f"{acc_blank:.3f}",
            "all_acc": f"{acc_all:.3f}"
        })


        if use_wandb and wandb is not None:
            wandb.log({
                "train/loss_ce": loss.item(),
                "train/acc_blank": acc_blank,
                "train/acc_all": acc_all,
                "lr": lr_now,
                "step": step
            })
    bar.close()
    return step

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    blank_id: int,
    max_batches: int | None = None,
    position=2  # TQDM stuff
    ) -> Tuple[float, float, float]:

    # sanity check
    if loader is None:
        return 0.0, 0.0, 0.0
    
    model.eval()
    losses, acc_blanks, acc_alls = [], [], []

    iterable = loader if max_batches is None else islice(loader, max_batches)
    bar = tqdm(iterable, desc="Eval", position=position, leave=False)
    for set_name, batch, _ in bar:
        inputs = batch['inputs'].to(device)
        labels = batch['labels'].to(device)

        logits = model(inputs)
        loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_LABEL_ID
            )
        
        # metrics and logging
        losses.append(loss.item())
        acc_b = blank_accuracy(logits, labels, inputs, blank_id=blank_id)
        acc_a = global_accuracy(logits, labels)
        acc_blanks.append(acc_b)
        acc_alls.append(acc_a)

        bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "blank_acc": f"{acc_b:.3f}",
            "all_acc": f"{acc_a:.3f}"
        })

    bar.close()
    n = max(1, len(losses))
    return sum(losses) / n, sum(acc_alls) / n, sum(acc_blanks) / n


# ---hyra---

@hydra.main(config_path="config", config_name="cfg_pretrain_baseline", version_base=None)
def main(cfg: Dict):
    set_seed(cfg.seed)
    device = get_device(cfg)
    print(f"[device] {device}")

    train_loader, train_meta = make_dataloader(cfg, split='train')
    eval_loader, eval_meta = (make_dataloader(cfg, split='test') if os.path.exists(
        os.path.join(cfg.data_paths[0], "test", "dataset.json")) else (None, train_meta))
    
    blank_id = cfg.blank_id
    model = build_model_from_cfg(cfg, train_meta, device) # model is on device

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2)
    )

    use_wandb = bool(getattr(cfg, "wandb", {}).get("enabled", False)) and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.get("project", "baseline"),
            name=getattr(cfg, "run_name", "run"),
            group = cfg.wandb.get("group", None),
            mode=cfg.wandb.get("mode", "online"),
            config=dict(cfg)
        )
        wandb.watch(model)
        print(f"[W&B] Logging enabled — run: {wandb.run.name if wandb.run else 'None'}")
    else:
        print("[W&B] Logging disabled.")

    steps_per_epoch = sum(1 for _ in iter(train_loader))
    total_steps = cfg.epochs * steps_per_epoch

    train_loader, _ = make_dataloader(cfg, split="train")

    step = 0
    master_bar = tqdm(range(cfg.epochs), desc="Training", position=0, leave=True)
    for epoch in master_bar:
        step = train_one_epoch(
            model, train_loader, device,
            blank_id=blank_id,
            base_lr=cfg.lr,
            warmup=cfg.lr_warmup_steps,
            total_steps=total_steps,
            min_ratio=cfg.lr_min_ratio,
            step0=step,
            opt=opt,
            use_wandb=use_wandb,
            position=1
        )
        # eval periodically to speedup runtime
        if (epoch + 1) % max(1, cfg.eval_interval) == 0 and eval_loader is not None:
            eval_loss, eval_acc_all, eval_acc_blank = evaluate(model, eval_loader, device, blank_id=blank_id, max_batches=cfg.max_eval_batches, position=2)
            if use_wandb:
                wandb.log({
                    "eval/loss_ce": eval_loss,
                    "eval/acc_all": eval_acc_all,
                    "eval/acc_blank": eval_acc_blank,
                    "epoch": epoch + 1
                })
            master_bar.set_postfix({
            "eval_epoch": f"{epoch+1}/{cfg.epochs}",
            "eval_loss": f"{eval_loss:.4f}",
            "eval_blank_acc": f"{eval_acc_blank:.3f}"
            })

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
    




