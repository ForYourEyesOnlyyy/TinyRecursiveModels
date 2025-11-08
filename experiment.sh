run_name="baseline_transformer_sudoku"

python baseline_train.py \
  arch=hrec_transformer \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=1000 \
  eval_interval=20 \
  lr=1e-3 \
  weight_decay=0.1 \
  global_batch_size=256 \
  +run_name=HRec6_6_compute_all