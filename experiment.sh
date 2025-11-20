run_name="baseline_transformer_sudoku"

python train.py \
  arch=hrec_transformer \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=1000 \
  eval_interval=20 \
  lr=1e-3 \
  weight_decay=0.1 \
  global_batch_size=128 \
  +run_name=Hrec15_3_alternative