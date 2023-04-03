python main.py \
    --blueprint blueprint/baseline-low-param-no-aug.yaml \
    --experiment-name baseline-low-param-no-aug-full-sample \
    --num-trial 5

python main.py \
    --blueprint blueprint/baseline-low-param-no-aug.yaml \
    --experiment-name baseline-low-param-no-aug-ten-percent-sample \
    --num-trial 5 \
    --sample-percentage 0.1 \
    --independent

python main.py \
  --blueprint blueprint/baseline-low-param-no-aug.yaml \
  --experiment-name baseline-low-param-no-aug-five-percent-sample \
  --num-trial 5 \
  --sample-percentage 0.05 \
  --independent

python main.py \
  --blueprint blueprint/baseline-low-param-no-aug.yaml \
  --experiment-name baseline-low-param-no-aug-one-percent-sample \
  --num-trial 5 \
  --sample-percentage 0.01 \
  --independent