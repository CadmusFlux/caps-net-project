rm -rf wandb

for blueprint in $(ls blueprint-ablation | grep baseline)
do
    experiment_name=${blueprint%%.*}
    python main.py \
        --blueprint blueprint-ablation/"$blueprint" \
        --experiment-name $experiment_name \
        --num-trial 5
done

for blueprint in $(ls blueprint-ablation | grep baseline)
do
    experiment_name=${blueprint%%.*}
    python main.py \
        --blueprint blueprint-ablation/"$blueprint" \
        --experiment-name $experiment_name-one-percent \
        --num-trial 5 \
        --sample-percentage 0.01 \
        --independent
done