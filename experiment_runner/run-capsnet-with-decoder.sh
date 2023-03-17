python main.py \
    --blueprint blueprint/capsnet-with-decoder-full-aug.yaml \
    --experiment-name capsnet-with-decoder-full-aug-full-sample \
    --num-trial 5 \
    --mixed-precision

python main.py \
    --blueprint blueprint/capsnet-with-decoder-full-aug.yaml \
    --experiment-name capsnet-with-decoder-full-aug-ten-percent-sample \
    --num-trial 5 \
    --mixed-precision \
    --sample-percentage 0.1 \
    --independent

python main.py \
  --blueprint blueprint/capsnet-with-decoder-full-aug.yaml \
  --experiment-name capsnet-with-decoder-full-aug-five-percent-sample \
  --num-trial 5 \
  --mixed-precision \
  --sample-percentage 0.05 \
  --independent

python main.py \
  --blueprint blueprint/capsnet-with-decoder-full-aug.yaml \
  --experiment-name baseline-full-aug-one-percent-sample \
  --num-trial 5 \
  --mixed-precision \
  --sample-percentage 0.01 \
  --independent