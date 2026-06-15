import torch
import time
import numpy as np
from omegaconf import OmegaConf
from mltau.tools.io.ParT_dataloader import ParTDataModule as ParTDataModuleRaw
from mltau.tools.io.preprocessed_ParTau_dataloader import ParTDataModule as ParTDataModulePre
from torch.utils.data import DataLoader

def benchmark_dataloader(name, dm_class, cfg, num_workers, n_batches=100):
    cfg.training.dataloader.num_dataloader_workers = num_workers
    
    start_init = time.time()
    dm = dm_class(cfg)
    dm.setup("test")
    loader = dm.test_dataloader()
    init_time = time.time() - start_init
    
    # Measure first batch
    start_first = time.time()
    iterator = iter(loader)
    try:
        batch = next(iterator)
        first_batch_time = time.time() - start_first
    except StopIteration:
        print(f"  [{name}] Error: Dataset empty?")
        return None

    # Measure subsequent batches
    start_throughput = time.time()
    count = 1 # already got one
    for _ in range(n_batches - 1):
        try:
            next(iterator)
            count += 1
        except StopIteration:
            break
    
    total_time = time.time() - start_throughput
    throughput = (count - 1) * cfg.training.dataloader.batch_size / total_time if count > 1 else 0
    
    print(f"  [{name}] Workers: {num_workers}")
    print(f"    Init time: {init_time:.4f}s")
    print(f"    First batch: {first_batch_time:.4f}s")
    print(f"    Throughput: {throughput:.2f} samples/s (over {count} batches)")
    
    return {
        "init_time": init_time,
        "first_batch_time": first_batch_time,
        "throughput": throughput
    }

def run_benchmarks():
    cfg = OmegaConf.create({
        "dataset": {
            "data_dir": "0509_dsinphi_to_sindphi",
            "max_cands": 20,
            "relative_sizes": {"train": 0.7, "val": 0.1, "test": 0.2}
        },
        "training": {
            "model": {"task": "is_tau", "name": "MultiParTau"},
            "dataloader": {
                "batch_size": 1024,
                "num_dataloader_workers": 0,
                "prefetch_factor": 2
            },
            "input_scaling": {"enabled": False}
        }
    })

    worker_counts = [0, 1, 4]
    results = {"Raw": {}, "Pre": {}}

    print("=== Benchmarking ParTDataModuleRaw (Parquet) ===")
    for nw in worker_counts:
        results["Raw"][nw] = benchmark_dataloader("Raw", ParTDataModuleRaw, cfg, nw)

    print("\n=== Benchmarking ParTDataModulePre (Preprocessed .pt) ===")
    for nw in worker_counts:
        results["Pre"][nw] = benchmark_dataloader("Pre", ParTDataModulePre, cfg, nw)

    # Summary table
    print("\n" + "="*50)
    print(f"{'Dataloader':<10} | {'Workers':<8} | {'Throughput (s/s)':<15}")
    print("-" * 50)
    for name in ["Raw", "Pre"]:
        for nw in worker_counts:
            tp = results[name][nw]["throughput"]
            print(f"{name:<10} | {nw:<8} | {tp:<15.2f}")
    print("="*50)

if __name__ == "__main__":
    run_benchmarks()
