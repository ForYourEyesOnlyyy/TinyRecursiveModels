python train.py \
  arch=hrec \
  arch.n_layers=3 \
  arch.recursion_steps=3 \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=2 \
  eval_interval=20 \
  lr=1e-3 \
  weight_decay=0.1 \
  global_batch_size=256 \
  wandb.group="sandbox" \