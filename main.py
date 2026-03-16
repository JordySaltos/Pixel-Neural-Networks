from Configuration import BaseConfig
from Loader import get_loader
from train import Solver


def create_dataloaders(config):
    train_loader = get_loader(
        config.dataset_dir,
        config.batch_size,
        train=True,
        dataset_name=config.dataset,
    )
    test_loader = get_loader(
        config.dataset_dir,
        config.batch_size,
        train=False,
        dataset_name=config.dataset,
    )
    return train_loader, test_loader


def run_training(config, train_loader, test_loader):
    solver = Solver(config, train_loader=train_loader, test_loader=test_loader)
    print(config)
    solver.build()
    solver.train()
    return solver


def main():
    """
    Entry point of the program. 
    Initializes configuration, loads data and starts training.
    """
    config = BaseConfig().initialize()
    train_loader, test_loader = create_dataloaders(config)
    run_training(config, train_loader, test_loader)


if __name__ == "__main__":
    main()