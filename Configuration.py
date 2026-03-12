import argparse
from datetime import datetime
from pathlib import Path
import pprint
import torch


# Define project directories
project_dir = Path(".").resolve()

datasets_dir = project_dir / "datasets"
datasets_dir.mkdir(exist_ok=True)

results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)


def get_optimizer(name="Adam"):
    """
    Returns a PyTorch optimizer class given its name.
    For example: "Adam" -> torch.optim.Adam
    """
    return getattr(torch.optim, name)


def str2bool(value):
    """
    Converts common string representations into boolean values.
    """

    value = value.lower()

    if value in ("yes", "true", "t", "y", "1"):
        return True

    if value in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


class BaseConfig(object):
    """
    Base configuration class.

    It handles:
    - parsing command line arguments
    - preparing dataset paths
    - creating directories for experiment results
    """

    def __init__(self):
        self._build_parser()

    def _build_parser(self):
        """
        Defines the command line arguments used by the project.
        """

        parser = argparse.ArgumentParser()

        # Mode of execution (train or test)
        parser.add_argument("--mode", type=str, default="train")

        # Training parameters
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_epochs", type=int, default=2)
        parser.add_argument("--optimizer", type=str, default="RMSprop")

        # Dataset name
        parser.add_argument("-dataset", type=str, default="CIFAR10")

        # Logging behaviour
        parser.add_argument("--log_interval", type=int, default=100)
        parser.add_argument("--save_interval", type=int, default=10)

        self.parser = parser

    def parse(self):
        """
        This method can be extended by subclasses
        if additional arguments are needed.
        """
        pass

    def initialize(self, parse=True, **extra_kwargs):
        """
        Parses arguments and prepares the experiment configuration.
        """

        self.parse()

        args = vars(self.parser.parse_known_args()[0])
        args.update(extra_kwargs)

        for key, value in args.items():

            if key == "optimizer":
                value = get_optimizer(value)

            setattr(self, key, value)

        self.isTrain = self.mode == "train"

        # Dataset directory
        self.dataset_dir = datasets_dir / self.dataset

        # Directory where experiment results will be stored
        self.model_dir = results_dir
        self.model_dir.mkdir(exist_ok=True)

        if self.mode == "train":

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.ckpt_dir = self.model_dir / timestamp
            self.ckpt_dir.mkdir()

            self._save_config()

        if self.mode == "test":
            assert self.load_ckpt_time
            self.ckpt_dir = self.model_dir / self.load_ckpt_time

        return self

    def _save_config(self):
        """
        Saves the configuration of the current experiment to a text file.
        """

        config_file = self.ckpt_dir / "config.txt"

        with open(config_file, "w") as f:

            f.write("Configurations\n")

            for key, value in sorted(self.__dict__.items()):
                f.write(f"{key}: {value}\n")

            f.write("End \n")

    def __repr__(self):
        """
        Pretty representation of the configuration.
        """

        return "Configurations\n" + pprint.pformat(self.__dict__)


def get_config(parse=True):
    """
    Helper function that creates and initializes the configuration
    in a single step.
    """

    return BaseConfig().initialize(parse=parse)