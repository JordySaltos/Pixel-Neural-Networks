
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Dataset-specific configurations
DATASET_CONFIGS = {
    "CIFAR10": {
        "n_channel": 3,
        "img_size": 32,
        "transform": transforms.Compose([transforms.ToTensor()]),
        "loader": datasets.CIFAR10,
    },
    "MNIST": {
        "n_channel": 1,
        "img_size": 32,   # padded from 28 to 32 so the model works identically
        "transform": transforms.Compose([
            transforms.Pad(2),          # 28x28 → 32x32
            transforms.ToTensor(),
        ]),
        "loader": datasets.MNIST,
    },
}


def get_loader(directory="./datasets",
               batch_size=128,
               train=True,
               dataset_name="CIFAR10",
               num_workers=1,
               pin_memory=True):
    """
    Creates a DataLoader for the chosen dataset (CIFAR10 or MNIST).

    Args:
        directory: path where the dataset is stored
        batch_size: number of images per batch
        train: whether to load the training or test set
        dataset_name: "CIFAR10" or "MNIST"
        num_workers: number of workers for data loading
        pin_memory: useful when training on GPU
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(DATASET_CONFIGS.keys())}."
        )

    cfg = DATASET_CONFIGS[dataset_name]

    dataset = cfg["loader"](
        root=directory,
        train=train,
        download=True,
        transform=cfg["transform"],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


def get_dataset_config(dataset_name: str) -> dict:
    """Returns the config dict (n_channel, img_size) for the given dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    return DATASET_CONFIGS[dataset_name]