# mini_loader.py
import multiprocessing as mp
from queue import Empty
import torch
from collections.abc import Mapping, Sequence

def default_collate(batch):
    # Handles tensors, tuples/lists, and dicts recursively.
    first = batch[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(batch, dim=0)
    if isinstance(first, Mapping):
        return {k: default_collate([d[k] for d in batch]) for k in first}
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        transposed = list(zip(*batch))
        return [default_collate(list(items)) for items in transposed]
    return batch

def _worker_loop(dataset, in_q, out_q, collate_fn):
    for task in iter(in_q.get, None):  # until None is sent
        seq, indices = task
        samples = [dataset[i] for i in indices]
        batch = collate_fn(samples)
        out_q.put((seq, batch))

class MiniDataLoader:
    def __init__(self, dataset, batch_size=1, num_workers=0,
                 collate_fn=default_collate, prefetch_factor=2, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        batches = [indices[i:i+self.batch_size]
                   for i in range(0, n, self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # Single-process: trivial
        if self.num_workers == 0:
            for b in batches:
                samples = [self.dataset[i] for i in b]
                yield self.collate_fn(samples)
            return

        # Multi-process
        in_q = mp.Queue()
        out_q = mp.Queue()
        workers = [mp.Process(target=_worker_loop,
                              args=(self.dataset, in_q, out_q, self.collate_fn))
                   for _ in range(self.num_workers)]
        for w in workers: w.start()

        # Enqueue all batches
        for seq, b in enumerate(batches):
            in_q.put((seq, b))

        # Collect in order
        results = {}
        expected = 0
        for _ in range(len(batches)):
            seq, batch = out_q.get()
            results[seq] = batch
            while expected in results:
                yield results.pop(expected)
                expected += 1

        # shutdown
        for _ in workers: in_q.put(None)
        for w in workers: w.join()
