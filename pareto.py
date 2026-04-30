"""
pareto_sweep_ddp.py
────────────────────
DDP-aware Pareto sweep for TRM-RTify.
Each rank processes its shard of the eval set; rank 0 all-reduces
the counts, computes metrics, and saves the results.

Launch with torchrun:
    torchrun --nproc_per_node=4 pareto.py \
        --checkpoint /root/thesis/me/TinyRecursiveModels/checkpoints/sudoku_baseline/TRM_RTify_long_phi10/best.ckpt \
        --data_dir   data/sudoku-extreme-2k-aug-1000 \
        --split      test \
        --theta_values 10.001 11 12 14 16 20 30 40 \
        --batch_size 768 \
        --output_dir results/pareto_full

    torchrun --nproc_per_node=4 pareto.py \
    --checkpoint /root/thesis/me/TinyRecursiveModels/checkpoints/sudoku_baseline/TRM_RTify_long_phi10/best.ckpt \
    --data_dir   data/sudoku-extreme-2k-aug-1000 \
    --batch_size 768 \
    --dry_run
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.recursive_reasoning.TRM_RTify import TRM_Rtify
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

IGNORE_LABEL_ID = -100


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    rank        = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def ddp_sum(value: float, device) -> float:
    """Sum a scalar across all ranks, return result on every rank."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = ckpt["config"]
    arch_cfg  = dict(train_cfg["arch"])

    arch_cfg["seq_len"]    = 81
    arch_cfg["vocab_size"] = 11

    model = TRM_Rtify(arch_cfg)

    # Strip torch.compile prefix
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    if is_rank0():
        epoch = ckpt.get("epoch", "?")
        step  = ckpt.get("step",  "?")
        score = ckpt.get("best_score") or 0.0
        print(f"[rank0] Loaded checkpoint  epoch={epoch}  step={step}  best_score={score:.4f}")
        print(f"[rank0] params={sum(p.numel() for p in model.parameters()):,}")

    return model, arch_cfg, train_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — each rank gets its own shard
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(data_dir: str, split: str, batch_size: int,
                 rank: int, world_size: int) -> tuple:
    ds_cfg = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[data_dir],
        global_batch_size=batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=rank,
        num_replicas=world_size,
    )
    dataset  = PuzzleDataset(ds_cfg, split=split)
    metadata = dataset.metadata
    loader   = DataLoader(
        dataset,
        batch_size=None,      # dataset already batches internally
        num_workers=1,        # IterableDataset: must be 1
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    # Estimate batches per rank for tqdm:
    # dataset yields ceil(total_puzzles / batch_size) batches total,
    # split evenly across world_size ranks
    n_batches = math.ceil(
        math.ceil(metadata.total_puzzles / batch_size)
    )
    return loader, metadata, n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Theta override
# ─────────────────────────────────────────────────────────────────────────────

def set_theta(model: TRM_Rtify, theta: float):
    theta_min = model.config.theta_min
    gap = theta - theta_min
    if gap <= 0:
        psi = -100.0
    else:
        psi = gap + math.log(-math.expm1(-gap))
    model.psi.data.fill_(psi)


# ─────────────────────────────────────────────────────────────────────────────
# Single eval pass at a given theta
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_at_theta(model: TRM_Rtify, loader: DataLoader,
                      theta: float, device: torch.device,
                      max_batches: int = None,
                      n_batches: int = None,
                      rank: int = 0) -> dict:

    original_psi = model.psi.data.clone()
    set_theta(model, theta)

    max_steps     = model.config.halt_max_steps
    local_exact   = 0.0
    local_steps   = 0.0
    local_puzzles = 0

    total = max_batches if max_batches is not None else n_batches
    # Only show tqdm on rank 0 to avoid cluttered multi-rank output
    pbar = tqdm(
        total=total,
        desc=f"θ={theta:.1f}",
        unit="batch",
        disable=(rank != 0),
        leave=False,
    )

    for i, (_, batch, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        batch  = {k: v.to(device).long() for k, v in batch.items()}
        labels = batch["labels"]
        B      = labels.shape[0]

        carry = model.initial_carry(batch)

        for _ in range(max_steps):
            carry, _ = model(carry, batch)
            if carry.halted.all():
                break

        preds      = carry.last_logits.argmax(dim=-1)
        blank_mask = labels != IGNORE_LABEL_ID
        exact      = ((preds == labels) | ~blank_mask).all(dim=-1).float().sum().item()

        local_exact   += exact
        local_steps   += carry.steps.float().sum().item()
        local_puzzles += B

        pbar.update(1)
        pbar.set_postfix(
            puzzles=int(local_puzzles),
            acc=f"{local_exact / local_puzzles:.1%}" if local_puzzles > 0 else "—",
        )

    pbar.close()

    # All-reduce across ranks
    total_exact   = ddp_sum(local_exact,   device)
    total_steps   = ddp_sum(local_steps,   device)
    total_puzzles = ddp_sum(local_puzzles, device)

    model.psi.data.copy_(original_psi)

    return {
        "theta":          theta,
        "exact_accuracy": total_exact   / total_puzzles,
        "mean_steps":     total_steps   / total_puzzles,
        "total_puzzles":  int(total_puzzles),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(results: list, output_path: str):
    steps    = [r["mean_steps"]     for r in results]
    accuracy = [r["exact_accuracy"] for r in results]
    thetas   = [r["theta"]          for r in results]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size":   12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(steps, accuracy, alpha=0.08, color="#2563eb")
    ax.plot(steps, accuracy, "-o", color="#2563eb", linewidth=2.2,
            markersize=8, markerfacecolor="white", markeredgewidth=2.2, zorder=3)

    # Theta labels alternating above/below
    for i, (s, a, t) in enumerate(zip(steps, accuracy, thetas)):
        dy = 12 if i % 2 == 0 else -16
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(f"θ={t:.0f}", xy=(s, a), xytext=(0, dy),
                    textcoords="offset points", fontsize=8,
                    color="#1e40af", ha="center", va=va)

    # TRM-ACT reference point and line
    ax.scatter([16], [0.72], marker="X", s=120, color="#dc2626",
               zorder=4, label="TRM-ACT (reproduced, no EMA)")
    ax.annotate("72.0%", xy=(16, 0.72), xytext=(-35, 8),
                textcoords="offset points", fontsize=9, color="#dc2626")
    ax.axvline(x=16, linestyle="--", linewidth=1.4, color="#dc2626", alpha=0.7)
    ax.text(16.2, 0.05, "TRM-ACT\n(fixed 16 steps)", color="#dc2626",
            fontsize=9, va="bottom", transform=ax.get_xaxis_transform())

    ax.set_xlabel("Mean reasoning steps at inference", fontsize=12)
    ax.set_ylabel("Exact accuracy", fontsize=12)
    ax.set_title("TRM-RTify: Speed–Accuracy Pareto Curve",
                 fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(left=3, right=18)
    ax.set_ylim(bottom=0.60, top=0.88)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"[rank0] Plot saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--split",        default="test")
    p.add_argument("--batch_size",   type=int,   default=768)
    p.add_argument("--output_dir",   default="results/pareto_full")
    p.add_argument("--theta_values", type=float, nargs="+",
                   default=[10.001, 11.0, 12.0, 14.0, 16.0, 20.0, 30.0, 40.0])
    p.add_argument("--dry_run",      action="store_true",
                   help="Run only 2 theta values on 2 batches per rank to verify the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_ddp()

    if is_rank0():
        os.makedirs(args.output_dir, exist_ok=True)
        mode = "DRY RUN" if args.dry_run else "FULL EVAL"
        print(f"[rank0] {mode}  world_size={world_size}  batch_size={args.batch_size}  split={args.split}")

    model, arch_cfg, train_cfg = load_model(args.checkpoint, device)
    loader, metadata, n_batches = build_loader(
        args.data_dir, args.split, args.batch_size, rank, world_size
    )

    if is_rank0():
        print(f"[rank0] dataset: {metadata.total_puzzles:,} puzzles  "
              f"~{n_batches} batches/rank")

    # Dry run: only 2 theta values, 2 batches each
    theta_values = sorted(args.theta_values)
    max_batches  = None
    if args.dry_run:
        theta_values = [theta_values[0], theta_values[-1]]
        max_batches  = 2
        if is_rank0():
            print(f"[rank0] DRY RUN: theta={theta_values}, max_batches={max_batches} per rank")

    results = []
    for theta in theta_values:
        if is_rank0():
            print(f"  theta={theta:6.3f} ...", end="  ", flush=True)

        r = evaluate_at_theta(
            model, loader, theta, device,
            max_batches=max_batches,
            n_batches=n_batches,
            rank=rank,
        )

        if is_rank0():
            results.append(r)
            print(f"exact_acc={r['exact_accuracy']:.4f}   "
                  f"mean_steps={r['mean_steps']:.2f}   "
                  f"puzzles={r['total_puzzles']:,}")

    if is_rank0():
        json_path = os.path.join(args.output_dir, "pareto_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[rank0] Results saved → {json_path}")

        plot_path = os.path.join(args.output_dir, "pareto_curve.png")
        plot_pareto(results, plot_path)

        print("\n── Summary ───────────────────────────────────────────")
        print(f"{'theta':>8}  {'steps':>8}  {'exact_acc':>10}  {'puzzles':>10}")
        print("-" * 44)
        for r in results:
            print(f"{r['theta']:>8.1f}  {r['mean_steps']:>8.2f}  "
                  f"{r['exact_accuracy']:>9.1%}  {r['total_puzzles']:>10,}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
