import argparse
from datetime import datetime
from pathlib import Path
import pprint
import torch


project_dir = Path(".").resolve()

datasets_dir = project_dir / "dataset"
datasets_dir.mkdir(exist_ok=True)

results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)


def get_optimizer(name="Adam"):
    """Returns a PyTorch optimizer class given its name."""
    return getattr(torch.optim, name)


def str2bool(value):
    """Converts common string representations into boolean values."""
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class BaseConfig(object):
    """
    Base configuration class.

    Handles parsing CLI arguments, preparing dataset paths,
    and creating directories for experiment results.
    """

    def __init__(self):
        self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--mode", type=str, default="train")

        # Training parameters
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_epochs", type=int, default=5)
        parser.add_argument("--optimizer", type=str, default="RMSprop")

        # Dataset
        parser.add_argument("--dataset", type=str, default="CIFAR10",
                            choices=["CIFAR10", "MNIST"])

        # Model architecture
        parser.add_argument("--h", type=int, default=128,
                            help="Bottleneck dimension of the PixelCNN")
        parser.add_argument("--n_block", type=int, default=15,
                            help="Number of residual blocks in the PixelCNN")
        parser.add_argument("--model_type", type=str, default="PixelCNN",
                            choices=["PixelCNN", "PixelRNN", "GatedPixelCNN"],
                            help="Autoregressive architecture to train")

        # Optimizer
        parser.add_argument("--lr", type=float, default=1e-3,
                            help="Initial learning rate")

        # Logging
        parser.add_argument("--log_interval", type=int, default=100)
        parser.add_argument("--save_interval", type=int, default=10)

        self.parser = parser

    def parse(self):
        pass

    def initialize(self, parse=True, **extra_kwargs):
        self.parse()

        args = vars(self.parser.parse_known_args()[0])
        args.update(extra_kwargs)

        for key, value in args.items():
            if key == "optimizer":
                value = get_optimizer(value)
            setattr(self, key, value)

        self.isTrain = self.mode == "train"
        self.dataset_dir = datasets_dir / self.dataset
        self.model_dir = results_dir
        self.model_dir.mkdir(exist_ok=True)

        if self.mode == "train":
            timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_type  = getattr(self, "model_type", "PixelCNN")
            folder_name = f"{model_type}_{self.dataset}_{timestamp}"
            self.ckpt_dir = self.model_dir / folder_name
            self.ckpt_dir.mkdir()
            self._save_config()

        if self.mode == "test":
            assert self.load_ckpt_time
            self.ckpt_dir = self.model_dir / self.load_ckpt_time

        return self

    def _save_config(self):
        config_file = self.ckpt_dir / "config.txt"
        with open(config_file, "w") as f:
            f.write("Configurations\n")
            for key, value in sorted(self.__dict__.items()):
                f.write(f"{key}: {value}\n")
            f.write("End\n")

    def __repr__(self):
        return "Configurations\n" + pprint.pformat(self.__dict__)


def get_config(parse=True):
    return BaseConfig().initialize(parse=parse)