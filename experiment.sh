python train.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=200 \
  eval_interval=20 \
  lr=1e-3 \
  weight_decay=0.1 \
  global_batch_size=512 \
  wandb.group="sandbox" \
  +run_name=test_job