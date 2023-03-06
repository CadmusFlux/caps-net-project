import argparse
import random

import joblib
import yaml

from utils.builder import *
from utils.dataloader import *
from utils.dataset import *

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
    "--random-seed",
    metavar="SEED",
    type=int,
    help="Set deterministic seed for reproducibility",
    default=0,
)

parser.add_argument(
    '--mixed-precision',
    type=bool,
    action='store_true',
    default=False
)


def preprocess_image(image: tf.Tensor):
    return dict(image=image / 255.0)


if __name__ == "__main__":
    args = parser.parse_args()

    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    with open(args.blueprint) as f:
        blueprint = yaml.safe_load(f)

        model_desc = blueprint["model"]
        train_args = blueprint["training_arguments"]

        basepath = pathlib.Path(f"{args.export_folder}/{args.experiment_name}")
        savepath = basepath
        for i in range(1, args.num_trials + 1):
            tf.keras.backend.clear_session()
            print(f"Trial {i}/{args.num_trials}")
            if args.num_trials > 1:
                savepath = pathlib.Path(basepath.joinpath(f"trial-{i}"))
                os.makedirs(savepath, exist_ok=True)

            model = tf.keras.Sequential(name=model_desc["name"])

            for name, options in model_desc["layer"].items():
                if not "hparams" in options:
                    options["hparams"] = {}
                model.add(
                    generate_layer(name, options["layer_type"], options["hparams"])
                )

            model.summary()

            trainset = CIFAR10(transform=preprocess_image)
            validset = CIFAR10(train=False, transform=preprocess_image)

            trainloader = DataLoader(trainset, train_args["batch_size"], True)
            validloader = DataLoader(validset, train_args["batch_size"])

            lr1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    train_args["batch_size"] * 1e-3 / 16, len(trainloader) * 100)

            optimizer = create_optimizer(lr1,train_args["optimizer"])
            loss = create_loss_function(train_args["loss"])
            metrics = train_args["metrics"]

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
            )

            history = model.fit(
                trainloader,
                validation_data=validloader,
                epochs=train_args["epochs"],
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        savepath.joinpath("weights/{epoch:03d}-{val_loss:.3f}.hdf5")
                    ),
                    tf.keras.callbacks.CSVLogger(savepath.joinpath("train.csv")),
                ],
            )

            joblib.dump(history.history, savepath.joinpath("train.pkl"))
