python main.py \
    --blueprint blueprint/efficient-capsnet-no-aug.yaml \
    --experiment-name efficient-capsnet-no-aug-ten-percent-sample \
    --num-trial 5 \
    --sample-percentage 0.1 \
    --independent

python main.py \
  --blueprint blueprint/efficient-capsnet-no-aug.yaml \
  --experiment-name efficient-capsnet-no-aug-five-percent-sample \
  --num-trial 5 \
  --sample-percentage 0.05 \
  --independent

python main.py \
  --blueprint blueprint/efficient-capsnet-no-aug.yaml \
  --experiment-name efficient-capsnet-no-aug-one-percent-sample \
  --num-trial 5 \
  --sample-percentage 0.01 \
  --independent