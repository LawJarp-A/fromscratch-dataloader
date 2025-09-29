import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from mini_loader import MiniDataLoader

def get_mnist(train: bool = True, data_root: str = "./data", retries: int = 1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return datasets.MNIST(root=data_root, train=train, transform=transform, download=True)
        except KeyboardInterrupt:
            raise
        except Exception as e: 
            last_err = e
            time.sleep(0.5)
    try:
        return datasets.MNIST(root=data_root, train=train, transform=transform, download=False)
    except Exception:
        if last_err is not None:
            raise SystemExit(f"Failed to prepare MNIST: {last_err}")
        raise SystemExit("Failed to prepare MNIST: unknown error")


def benchmark(loader, name: str, max_batches: int = 100) -> None:
    start = time.time()
    try:
        for i, batch in enumerate(loader):
            _ = batch
            if i >= max_batches:
                break
    except KeyboardInterrupt:
        print(f"Interrupted: {name}")
    finally:
        dur = time.time() - start
        print(f"{name}: {dur:.4f} sec (~{max_batches} batches)")


if __name__ == "__main__":
    train_ds = get_mnist(train=True, retries=2)

    torch_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    mini_dl = MiniDataLoader(train_ds, batch_size=128, num_workers=2)

    benchmark(torch_dl, "torch.DataLoader (MNIST)")
    benchmark(mini_dl, "MiniDataLoader (MNIST)")


