import argparse
from datetime import datetime
from pathlib import Path
import pprint
import torch


project_dir = Path(".").resolve()

datasets_dir = project_dir / "dataset"
results_dir  = project_dir / "results"


def get_optimizer(name="Adam"):
    """
    Return a PyTorch optimizer class by its name.

    Args:
        name (str): Name of the optimizer (e.g., "Adam", "RMSprop").

    Returns:
        torch.optim.Optimizer: Corresponding PyTorch optimizer class.
    """
    return getattr(torch.optim, name)

class BaseConfig(object):
    """
    Base configuration class for experiments.

    Handles parsing command-line arguments, setting up dataset paths,
    creating experiment directories, and saving configuration files.
    """

    def __init__(self):
        """Initialize the BaseConfig by building the argument parser."""
        self._build_parser()

    def _build_parser(self) -> None:
        """Build the argparse parser with default experiment parameters."""
        parser = argparse.ArgumentParser()

        parser.add_argument("--mode", type=str, default="train")

        # Training parameters
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--n_epochs", type=int, default=40)

        # Dataset
        parser.add_argument("--dataset", type=str, default="CIFAR10",
                            choices=["CIFAR10", "MNIST"])

        # Model architecture
        parser.add_argument("--h", type=int, default=128,
                            help="Bottleneck dimension of the PixelCNN")
        parser.add_argument("--n_block", type=int, default=10,
                            help="Number of residual blocks in the PixelCNN")
        parser.add_argument("--model_type", type=str, default="PixelCNN",
                            choices=["PixelCNN", "PixelRNN", "GatedPixelCNN"],
                            help="Autoregressive architecture to train")

        # Optimizer
        parser.add_argument("--lr", type=float, default=1e-3,
                            help="Initial learning rate")
        parser.add_argument(
            "--optimizer", type=str, default="Adam",
            choices=["Adam", "AdamW", "RMSprop"],
            help="Optimiser class",
        )

        # Logging
        parser.add_argument("--log_interval", type=int, default=100)
        parser.add_argument("--save_interval", type=int, default=10)

        self.parser = parser

    def parse(self) -> None:
        """
        Hook method for subclasses to add extra arguments before parsing.

        Can be overridden by subclasses if additional CLI arguments
        are needed.
        """
        pass

    def initialize(self, parse=True, **extra_kwargs):
        """
        Parse CLI arguments, override with extra kwargs, and set up directories.

        Args:
            parse (bool): Whether to parse command-line arguments.
                Set False when using this class in a GUI (e.g., Streamlit)
                to avoid argparse conflicts.
            **extra_kwargs: Keyword arguments that override parsed CLI args
                (e.g., dataset="MNIST", lr=0.001).

        Returns:
            BaseConfig: Self, fully initialized and ready for use.
        """
        self.parse()

        args = vars(self.parser.parse_known_args()[0])
        args.update(extra_kwargs)

        for key, value in args.items():
            if key == "optimizer":
                value = get_optimizer(value)
            setattr(self, key, value)

        datasets_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)

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

    def _save_config(self) -> None:
        """
        Save all configuration attributes to a 'config.txt' file
        in the checkpoint directory.
        """
        config_file = self.ckpt_dir / "config.txt"
        with open(config_file, "w") as f:
            f.write("Configurations\n")
            for key, value in sorted(self.__dict__.items()):
                f.write(f"{key}: {value}\n")
            f.write("End\n")

    def __repr__(self) -> str:
        """
        Return a human-readable string representation of the configuration.

        Returns:
            str: Pretty-printed configuration dictionary.
        """
        return "Configurations\n" + pprint.pformat(self.__dict__)