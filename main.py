Config = BaseConfig


def create_dataloaders(config):
    """
    Creates the training and test dataloaders.
    """

    train_loader = get_loader(
        config.dataset_dir,
        config.batch_size,
        train=True
    )

    test_loader = get_loader(
        config.dataset_dir,
        config.batch_size,
        train=False
    )

    return train_loader, test_loader


def run_training(config, train_loader, test_loader):
    """
    Initializes the solver and runs the training procedure.
    """

    solver = Solver(
        config,
        train_loader=train_loader,
        test_loader=test_loader
    )

    print(config)

    solver.build()
    solver.train()


def main():
    """
    Entry point of the program.
    Initializes configuration, loads data and starts training.
    """

    config = Config().initialize()

    train_loader, test_loader = create_dataloaders(config)

    run_training(config, train_loader, test_loader)


if __name__ == "__main__":
    main()