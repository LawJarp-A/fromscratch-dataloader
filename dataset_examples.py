from typing import Any, Tuple


try:
    from torchvision import datasets, transforms
except Exception:  # optional in some environments
    datasets = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]


def get_mnist_datasets(data_dir: str = "./data", download: bool = True) -> Tuple[Any, Any]:
    """Return (train, test) MNIST datasets with standard normalization.

    Requires torchvision. Images are normalized with mean=0.1307, std=0.3081.
    """
    if datasets is None or transforms is None:
        raise ImportError("torchvision is required for MNIST. Please install torchvision.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    test = datasets.MNIST(root=data_dir, train=False, transform=transform, download=download)
    return train, test


__all__ = [
    "get_mnist_datasets",
]


