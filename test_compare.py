import time
import torch
from torch.utils.data import DataLoader
from mini_loader import MiniDataLoader

class ToyDataset:
    def __init__(self, n): self.data = torch.arange(n)
    def __getitem__(self, i): return self.data[i]
    def __len__(self): return len(self.data)

def benchmark(loader, name):
    start = time.time()
    for batch in loader:
        _ = batch  # simulate training step
    dur = time.time() - start
    print(f"{name}: {dur:.4f} sec")

if __name__ == "__main__":
    ds = ToyDataset(100000)

    torch_dl = DataLoader(ds, batch_size=64, num_workers=2)
    mini_dl  = MiniDataLoader(ds, batch_size=64, num_workers=2)

    benchmark(torch_dl, "torch.DataLoader")
    benchmark(mini_dl,  "MiniDataLoader")