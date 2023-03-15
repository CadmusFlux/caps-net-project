import argparse
import random

import yaml
from wandb.keras import WandbMetricsLogger

import wandb
from utils.builder import *
from utils.dataloader import *
from utils.dataset import *
from utils.model import *

parser = argparse.ArgumentParser(
    prog="CapsNet-Experiment", description="Run experiments"
)

parser.add_argument(
    "--blueprint",
    metavar="YAML",
    type=str,
    help="YAML file for training description.",
    required=True,
)

parser.add_argument(
    "--experiment-name",
    metavar="EXPNAME",
    type=str,
    help="Experiment name and export.",
    required=True,
)

parser.add_argument(
    "--export-folder",
    metavar="FOLDERNAME",
    type=str,
    help="Base folder for exporting experiment.",
    default="experiment",
)

parser.add_argument(
    "--num-trials", metavar="N", type=int, help="Number of runs for training", default=1
)

parser.add_argument(
    "--sample-percentage",
    metavar="P",
    type=float,
    help="Percentage of sampled training data",
    default=1.0,
)

parser.add_argument(
    "--sample-size",
    metavar="N",
    type=int,
    help="Fixed number of sampled training data",
    default=-1,
)

parser.add_argument(
    "--random-seed",
    metavar="SEED",
    type=int,
    help="Set deterministic seed for reproducibility",
    default=0,
)

parser.add_argument(
    "--independent",
    action=argparse.BooleanOptionalAction,
    help="Use independent holdout sampling to have disjoint split per trial",
)

parser.add_argument(
    "--mixed-precision",
    action=argparse.BooleanOptionalAction,
    help="Use mixed precision training",
)

if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    sample_size = args.sample_size
    if sample_size < 0:
        sample_size = args.sample_percentage

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    with open(args.blueprint) as f:
        blueprint = yaml.safe_load(f)

        dataset = blueprint["dataset"]
        model_desc = blueprint["model"]
        train_args = blueprint["training_arguments"]

        basepath = pathlib.Path(f"{args.export_folder}/{args.experiment_name}")
        savepath = basepath

        os.makedirs(basepath, exist_ok=True)

        if 'augmentation' not in train_args:
            train_args['augmentation'] = []
        aug_tag = ' '.join(train_args['augmentation'])

        transform_train = create_transform(dataset, train_args["augmentation"])
        transform_eval = create_transform(dataset)

        if dataset.upper() == "CIFAR10":
            dataset = CIFAR10("train", transform_train)
            dataset_valid = CIFAR10("valid", transform_eval)
        elif dataset.upper() == "MNIST":
            dataset = MNIST("train", transform_train)
            dataset_valid = MNIST("valid", transform_eval)

        dataset = dataset.sample(sample_size, args.num_trials, args.independent)

        project_name = args.experiment_name
        for i, dataset_train in enumerate(dataset, 1):
            tf.keras.backend.clear_session()
            print(f"Trial {i}/{args.num_trials}")
            if args.num_trials > 1:
                savepath = pathlib.Path(basepath.joinpath(f"trial-{i}"))
                os.makedirs(savepath, exist_ok=True)
                project_name = f'{args.experiment_name}-trial-{i}'

            tags = [model_desc['name'], model_desc['kind']]

            if len(aug_tag) > 0:
                tags.append(aug_tag)

            wandb.init(project='mlp-cw', name=project_name, config=blueprint, job_type='training',
                       tags=tags, reinit=True)

            model = create_model(model_desc, dataset_valid.image_size, dataset_valid.num_classes)

            trainloader = DataLoader(dataset_train, train_args["batch_size"], True)
            validloader = DataLoader(dataset_valid, train_args["batch_size"])

            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                train_args["batch_size"] * float(train_args["base_learning_rate"]) / 16,
                len(trainloader) * train_args["epochs"],
            )

            optimizer = create_optimizer(learning_rate, train_args["optimizer"])

            model.compile(optimizer)

            model.summary()

            history = model.fit(
                trainloader,
                validation_data=validloader,
                epochs=train_args["epochs"],
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        savepath.joinpath("weights.hdf5"),
                        save_best_only=True,
                        save_weights_only=True,
                    ),
                    tf.keras.callbacks.TerminateOnNaN(),
                    tf.keras.callbacks.CSVLogger(savepath.joinpath("train.csv")),
                    WandbMetricsLogger()
                ],
            )
