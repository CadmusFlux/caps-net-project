python main.py \
    --blueprint blueprint/resnet50-full-aug.yaml \
    --experiment-name resnet50-full-aug-full-sample \
    --num-trial 5

python main.py \
    --blueprint blueprint/resnet50-full-aug.yaml \
    --experiment-name resnet50-full-aug-ten-percent-sample \
    --num-trial 5 \
    --sample-percentage 0.1 \
    --independent

python main.py \
  --blueprint blueprint/resnet50-full-aug.yaml \
  --experiment-name resnet50-full-aug-five-percent-sample \
  --num-trial 5 \
  --sample-percentage 0.05 \
  --independent

python main.py \
  --blueprint blueprint/resnet50-full-aug.yaml \
  --experiment-name resnet50-full-aug-one-percent-sample \
  --num-trial 5 \
  --sample-percentage 0.01 \
  --independent