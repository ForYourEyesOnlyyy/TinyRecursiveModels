from typing import List, Any

def build_run_name(cfg: Any) -> str:
    """
    Build a descriptive run name string based on the architecture type
    and its key hyperparameters.

    Supports:
      - SudokuTransformer (baseline)
      - HRecTransformer (recursive)
      - Fallback to "UNK_ARCH" if architecture is unrecognized.

    Returns
    -------
    str
        A human-readable, wandb-safe run name.
    """
    arch_name = getattr(cfg.arch, "name", None)
    parts: List[str] = []

    # Extract class suffix if name="module@Class"
    if arch_name:
        arch_class = arch_name.split("@")[-1]
    else:
        arch_class = None

    # ---------- Baseline Transformer ----------
    if arch_class == "BasicTransformer":
        parts.append("Baseline")
        n_layers = getattr(cfg.arch, "n_layers", None)
        if n_layers is not None:
            parts.append(f"L{n_layers}")

    # ---------- Recursive Transformer ----------
    elif arch_class == "HRecTransformer":
        parts.append("HRec")
        n_layers = getattr(cfg.arch, "n_layers", None)
        rec_steps = getattr(cfg.arch, "recursion_steps", None)
        tbptt = getattr(cfg.arch, "detach_till_last", None)
        ds = getattr(cfg.arch, "deep_supervision", None)
        ru = getattr(cfg.arch, "residual_update", None)

        if n_layers is not None:
            parts.append(f"L{n_layers}")
        if rec_steps is not None:
            parts.append(f"T{rec_steps}")
        if tbptt is not None:
            parts.append(f"TBPTT={tbptt}")
        if ds is not None:
            parts.append(f"DS={ds}")
        if ru is not None:
            parts.append(f"RU={ru}")

    # ---------- Unknown Arch ----------
    else:
        parts.append("UNK_ARCH")

    return "_".join(parts)

def build_arch_tags(cfg) -> list[str]:
    """
    Given a train config object (with cfg.arch),
    construct a list of descriptive W&B tags based on architecture parameters.
    Supports both baseline and recurrent transformer configs.

    Expected fields inside cfg.arch (optional):
        - name: str
        - n_layers: int
        - recursion_steps: int
        - detach_till_last: bool
        - deep_supervision: bool
        - residual_update: bool

    Returns:
        List of descriptive tags.
    """

    # Base model name (e.g. "HRecTransformer")
    arch_name = getattr(cfg.arch, "name", None)
    base_name = arch_name.split("@")[-1] if arch_name else "Model"

    # Optional fields — may not exist for baseline models
    n_layers  = getattr(cfg.arch, "n_layers", None)
    rec_steps = getattr(cfg.arch, "recursion_steps", None)
    tbptt     = getattr(cfg.arch, "detach_till_last", None)
    ds        = getattr(cfg.arch, "deep_supervision", None)
    ru        = getattr(cfg.arch, "residual_update", None)

    tags = [base_name]

    if n_layers is not None:
        tags.append(f"L={n_layers}")

    if rec_steps is not None:
        tags.append(f"T={rec_steps}")

    if tbptt is not None:
        tags.append("TBPTT" if tbptt else "FullBPTT")

    if ds is not None:
        tags.append("DS" if ds else "noDS")

    if ru is not None:
        tags.append("RU" if ru else "noRU")

    return tags