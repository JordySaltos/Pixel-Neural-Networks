"""
CLI entry point for training a Pixel Neural Network.

Usage::

    python main.py --model_type GatedPixelCNN --dataset MNIST --n_epochs 20

All arguments are parsed by :class:`Configuration.BaseConfig`.
"""

from pathlib import Path
from Configuration import BaseConfig
from Loader import get_loader
from train import Solver


def create_dataloaders(config):
    """
    Build train and test DataLoaders from a BaseConfig.

    Args:
        config: An initialised BaseConfig instance

    Returns:
        tuple: (train_loader, test_loader)
    """
    dataset_dir = Path(".") / "dataset" / config.dataset
    train_loader = get_loader(
        dataset_dir,
        config.batch_size,
        train=True,
        dataset_name=config.dataset,
    )
    test_loader = get_loader(
        dataset_dir,
        config.batch_size,
        train=False,
        dataset_name=config.dataset,
    )
    return train_loader, test_loader


def run_training(config, train_loader, test_loader):
    """
    Instantiate a Solver, build the model, and run the training loop.

    Args:
        config: An initialised BaseConfig instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation/test data

    Returns:
        Solver: The trained Solver instance
    """
    solver = Solver(config, train_loader=train_loader, test_loader=test_loader)
    print(config)
    print(f"Model type: {getattr(config, 'model_type', 'PixelCNN')}")
    solver.build()
    solver.train()
    return solver


def main() -> None:
    """
    Parse CLI arguments, build data loaders, and start the training loop.

    Returns:
        None
    """
    config = BaseConfig().initialize()
    train_loader, test_loader = create_dataloaders(config)
    run_training(config, train_loader, test_loader)


if __name__ == "__main__":
    main()