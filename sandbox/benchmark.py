import time
import torch
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utilities.data.dataloader import get_dataloader
from utilities.os_utilities import load_configuration
from sandbox.fast_dataloader import get_fast_dataloader

def benchmark():
    data_folder = "data_examples/openfold"
    split_file = "dataset_splits/Train_small.json"
    config_file = "configurations/tiny_configuration.yaml"
    precomputed_dir = "sandbox/precomputed_data"
    
    config = load_configuration(config_file)
    device = torch.device("cpu")
    dtype = torch.float32

    print("--- Starting Benchmark ---")

    # 1. Benchmark Original Dataloader
    # Note: We include the initialization time as it's part of the overhead
    start_time = time.time()
    original_dataloader = get_dataloader(
        data_folder=data_folder,
        model_configuration=config,
        split_path=split_file,
        phase="Train",
        device=device,
        dtype=dtype,
        num_workers=0
    )
    
    print("Iterating through original dataloader...")
    count = 0
    for batch in original_dataloader:
        count += 1
    original_duration = time.time() - start_time
    print(f"Original Dataloader: {original_duration:.4f} seconds for {count} batches.")

    # 2. Benchmark Fast Dataloader
    start_time = time.time()
    fast_dataloader = get_fast_dataloader(
        precomputed_dir=precomputed_dir,
        split_file_path=split_file,
        number_samples=5,
        batch_size=config["TrainDataConfiguration"]["batch_size"],
        num_workers=0,
        shuffle=False
    )
    
    print("Iterating through fast dataloader...")
    count = 0
    for batch in fast_dataloader:
        count += 1
    fast_duration = time.time() - start_time
    print(f"Fast Dataloader: {fast_duration:.4f} seconds for {count} batches.")

    # 3. Summary
    speedup = original_duration / fast_duration if fast_duration > 0 else 0
    print(f"\nSummary: The fast dataloader is {speedup:.2f}x faster.")

if __name__ == "__main__":
    benchmark()
