python train.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  global_batch_size=512 \
  epochs=5000 \
  n_reasoning_episodes=16 \
  arch.S_steps=3 \
  arch.R_steps=6 \
  arch.n_layers=2 \
  arch.scale_input_injection=False \
  wandb.group="TRM_NoACT" \
  +run_name=TRM_NoACT_8ep_S3_R6_B2_stable_128b_no_gating