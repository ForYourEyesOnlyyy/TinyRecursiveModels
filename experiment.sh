python train_distributed.py \
  device=mps \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  global_batch_size=768 \
  epochs=2000 \
  eval_interval=50 \
  n_reasoning_episodes=1 \
  arch.S_steps=3 \
  arch.R_steps=6 \
  arch.n_layers=2 \
  arch.scale_input_injection=False \
  wandb.group="TRM_NoACT_clean" \