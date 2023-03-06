python main.py \
  --blueprint blueprint/baseline.yaml \
  --experiment-name cifar10-baseline-sample-one-percent \
  --num-trials 3 \
  --sample-percentage 0.01 \

python main.py \
  --blueprint blueprint/vit-small.yaml \
  --experiment-name cifar10-vit-small-sample-ten-percent \
  --num-trials 3 \
  --sample-percentage 0.1 \
  --independent

