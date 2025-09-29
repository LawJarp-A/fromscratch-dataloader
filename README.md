# fromscratch-dataloader

*A minimal reimplementation of PyTorch’s DataLoader*

Inspired by [Andrej Karpathy’s](https://github.com/karpathy) style of projects like `nanoGPT`, this repo strips down PyTorch’s [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to its essence.

## Why?

PyTorch’s DataLoader is **powerful but complex** (~1600 lines of code). At its core, it’s just:
- batching indices
- fetching samples
- collating into tensors
- parallelizing + prefetching

This repo makes it **simple and readable (~60 lines)**

## What’s inside?

- `mini_loader.py` → Minimal DataLoader (Python, ~60 lines)
- `dataset_examples.py` → Tiny toy datasets
- `test_compare.py` → Compare against `torch.utils.data.DataLoader` for speed
- `docs/theory.md` → Explanation of how DataLoader works
- `rust/` →  Work-in-progress Rust reimplementation

## Quickstart

```bash
git clone https://github.com/<you>/fromscratch-dataloader.git
cd fromscratch-dataloader
pip install -r requirements.txt
