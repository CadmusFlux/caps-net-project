import argparse
import random

import yaml
from keras_cv import models

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
    action="store_true",
    help="Use independent holdout sampling to have disjoint split per trial",
    default=False,
)

parser.add_argument(
    "--mixed-precision",
    action="store_true",
    help="Use mixed precision training",
    default=False,
)

if __name__ == "__main__":
    args = parser.parse_args()

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

        if "augmentation" not in train_args:
            train_args["augmentation"] = None

        basepath = pathlib.Path(f"{args.export_folder}/{args.experiment_name}")
        savepath = basepath

        os.makedirs(basepath, exist_ok=True)

        transform_train = create_transform(dataset, train_args["augmentation"])
        transform_eval = create_transform(dataset)

        if dataset.upper() == "CIFAR10":
            dataset = CIFAR10("train", transform_train)
            dataset_valid = CIFAR10("valid", transform_eval)
        elif dataset.upper() == "MNIST":
            dataset = MNIST("train", transform_train)
            dataset_valid = MNIST("valid", transform_eval)

        dataset = dataset.sample(sample_size, args.num_trials, args.independent)

        for i, dataset_train in enumerate(dataset, 1):
            tf.keras.backend.clear_session()
            print(f"Trial {i}/{args.num_trials}")
            if args.num_trials > 1:
                savepath = pathlib.Path(basepath.joinpath(f"trial-{i}"))
                os.makedirs(savepath, exist_ok=True)

            model = tf.keras.Sequential(name=model_desc["name"])

            if model_desc["kind"] in ["standard", "capsule_network"]:
                pretrained_args = dict(
                    include_rescaling=False,
                    include_top=True,
                    input_shape=dataset_valid.image_size,
                    classes=dataset_valid.num_classes,
                    classifier_activation="linear",
                )
                if model_desc["name"] == "resnet18":
                    backbone = models.resnet_v2.ResNet18V2(**pretrained_args)
                elif model_desc["name"] == "resnet34":
                    backbone = models.resnet_v2.ResNet34V2(**pretrained_args)
                elif model_desc["name"] == "resnet50":
                    backbone = models.resnet_v2.ResNet50V2(**pretrained_args)
                elif model_desc["name"] == "densenet121":
                    backbone = models.densenet.DenseNet121(**pretrained_args)
                elif model_desc["name"] == "densenet169":
                    backbone = models.densenet.DenseNet169(**pretrained_args)
                elif model_desc["name"] == "vit-tiny":
                    backbone = models.vit.ViTTiny16(**pretrained_args)
                elif model_desc["name"] == "vit-small":
                    backbone = models.vit.ViTS16(**pretrained_args)
                elif model_desc["name"] == "vit-base":
                    backbone = models.vit.ViTB16(**pretrained_args)
                else:
                    backbone = tf.keras.Sequential(name=model_desc["name"])
                    for name, options in model_desc["layer"].items():
                        if "hparams" not in options:
                            options["hparams"] = {}
                        backbone.add(
                            create_layer(
                                name, options["layer_type"], options["hparams"]
                            )
                        )

                if model_desc["kind"] == "standard":
                    model = StandardModel(backbone)
                else:
                    model = CapsNet(backbone)

            else:
                encoder = tf.keras.Sequential(name=model_desc["encoder"]["name"])
                decoder = tf.keras.Sequential(name=model_desc["decoder"]["name"])

                for name, options in model_desc["encoder"]["layer"].items():
                    if "hparams" not in options:
                        options["hparams"] = {}
                    encoder.add(
                        create_layer(name, options["layer_type"], options["hparams"])
                    )

                for name, options in model_desc["decoder"]["layer"].items():
                    if "hparams" not in options:
                        options["hparams"] = {}
                    decoder.add(
                        create_layer(name, options["layer_type"], options["hparams"])
                    )

                model = CapsNetWithDecoder(encoder, decoder)

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
                ],
            )
