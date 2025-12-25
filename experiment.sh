python train.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
<<<<<<< Updated upstream
  epochs=200 \
  eval_interval=20 \
=======
  epochs=100 \
  eval_interval=10 \
>>>>>>> Stashed changes
  lr=1e-3 \
  weight_decay=0.1 \
  global_batch_size=512 \
  wandb.group="sandbox" \
  +run_name=test_job