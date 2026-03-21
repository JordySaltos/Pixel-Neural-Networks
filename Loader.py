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


def get_loader(directory="./dataset",
               batch_size=128,
               train=True,
               dataset_name="CIFAR10",
               num_workers=0,
               pin_memory=True):
    """
    Create a PyTorch DataLoader for a given dataset.

    Args:
        directory (str): Root path where datasets are stored. The dataset
            loader will create a subfolder with the dataset name automatically.
        batch_size (int): Number of images per batch.
        train (bool): Whether to load the training split (True) or test split (False).
        dataset_name (str): Name of the dataset to load ("CIFAR10" or "MNIST").
        num_workers (int): Number of subprocesses to use for data loading.
            Defaults to 0 (useful for Windows).
        pin_memory (bool): If True, the data loader will copy tensors into CUDA
            pinned memory before returning them. Useful when training on GPU.

    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    import os
    from pathlib import Path

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from {list(DATASET_CONFIGS.keys())}."
        )

    cfg = DATASET_CONFIGS[dataset_name]

    root = Path(directory)
    if root.name == dataset_name:
        root = root.parent

    dataset = cfg["loader"](
        root=str(root),
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
    """
    Retrieve configuration parameters for a dataset.

    Args:
        dataset_name (str): Name of the dataset ("CIFAR10" or "MNIST").

    Returns:
        dict: Dictionary containing dataset parameters:
            - 'n_channel': number of channels in the images
            - 'img_size': image height and width
            - 'transform': torchvision transform applied to the dataset
            - 'loader': torchvision dataset class

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    return DATASET_CONFIGS[dataset_name]