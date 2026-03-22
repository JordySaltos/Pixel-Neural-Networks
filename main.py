"""
CLI entry point for training a Pixel Neural Network.

Usage::

    python main.py --model_type GatedPixelCNN --dataset MNIST --n_epochs 20

All arguments are parsed by :class:`Configuration.BaseConfig`.
"""

from Configuration import BaseConfig
from train import Solver, GatedSolver, SOLVER_REGISTRY, build_data_loaders


def run_training(config, train_loader, test_loader):
    """
    Instantiate a Solver, build the model, and run the training loop.

    Args:
        config (BaseConfig): An initialised BaseConfig instance.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation/test data.

    Returns:
        Solver: The trained Solver instance.
    """
    solver_class = SOLVER_REGISTRY.get(
        getattr(config, "model_type", "PixelCNN"), "Solver"
    )
    solver_class = GatedSolver if solver_class == "GatedSolver" else Solver
    solver = solver_class(config, train_loader=train_loader, test_loader=test_loader)
    print(config)
    print(f"Model type: {getattr(config, 'model_type', 'PixelCNN')}")
    solver.build()
    solver.train()
    return solver


def main() -> None:
    """
    Parse CLI arguments, build data loaders, and start the training loop.
    """
    config = BaseConfig().initialize()
    train_loader, test_loader = build_data_loaders(
        config.dataset, config.batch_size
    )
    run_training(config, train_loader, test_loader)


if __name__ == "__main__":
    main()