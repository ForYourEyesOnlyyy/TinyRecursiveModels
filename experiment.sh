run_name="baseline_transformer_sudoku"

python baseline_train.py \
  arch=transformers_baseline \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=5000 \
  eval_interval=100 \
  lr=1e-3 \
  weight_decay=0.1 \
  global_batch_size=256 \
  +run_name=layers_15_big