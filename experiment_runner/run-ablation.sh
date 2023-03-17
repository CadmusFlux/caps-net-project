rm -rf wandb

for blueprint in $(ls blueprint-ablation)
do
    experiment_name=${blueprint%%.*}
    python main.py \
        --blueprint blueprint-ablation/"$blueprint" \
        --experiment-name $experiment_name \
        --num-trial 5 \
        --mixed-precision
done
