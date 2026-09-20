import os
import glob
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_val_loss(info):
    dir_path, dir_name, event_file, ds_size_fallback, m_size = info
    print(f"  Processing {dir_name}...")
    
    # Try to get actual dataset size from dataset_size.txt
    ds_size = ds_size_fallback
    ds_size_file = os.path.join(dir_path, "dataset_size.txt")
    if os.path.exists(ds_size_file):
        try:
            with open(ds_size_file, "r") as f:
                for line in f:
                    if line.startswith("total:"):
                        ds_size = int(line.split(":")[1].strip())
                        break
        except Exception as e:
            print(f"    {dir_name}: Error reading dataset_size.txt: {e}")

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        if 'val_losses/loss' in ea.Tags()['scalars']:
            events = ea.Scalars('val_losses/loss')
            v_min = min([e.value for e in events])
            print(f"    {dir_name}: Found val_loss: {v_min:.4f} (ds_size: {ds_size})")
            return {
                "model_size": m_size,
                "dataset_size": ds_size,
                "val_loss": v_min,
                "dir": dir_name
            }
        else:
            print(f"    {dir_name}: val_losses/loss not found.")
    except Exception as e:
        print(f"    {dir_name}: Error processing: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract scaling study results.")
    parser.add_argument("input_dir", help="Directory containing scaling study runs (e.g. outputs/scaling_0611)")
    parser.add_argument("--output", default="scaling_results.csv", help="Output CSV file")
    args = parser.parse_args()

    output_base = args.input_dir
    if not os.path.exists(output_base):
        print(f"Error: Directory {output_base} does not exist.")
        return

    dirs = [d for d in os.listdir(output_base) if d.startswith("mp_")]
    print(f"Found {len(dirs)} candidate directories in {output_base}")

    tasks = []
    for d in dirs:
        parts = d.split("_")
        if len(parts) < 3: continue
        
        ds_size_str = parts[1][2:] # strip "ds"
        m_size = parts[2][1:]      # strip "m"
        
        if ds_size_str == "null":
            ds_size_fallback = 2000000 
        else:
            ds_size_fallback = int(ds_size_str)
            
        full_dir_path = os.path.join(output_base, d)
        path = os.path.join(full_dir_path, "tensorboard/ParTau_experiment/version_0")
        event_files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        
        if event_files:
            tasks.append((full_dir_path, d, event_files[0], ds_size_fallback, m_size))
        else:
            print(f"  No event files found in {path}")

    results = []
    print(f"Starting parallel extraction with {min(len(tasks), os.cpu_count())} workers...")
    with ProcessPoolExecutor() as executor:
        for res in executor.map(extract_val_loss, tasks):
            if res:
                results.append(res)

    print(f"Extracted {len(results)} results. Saving to {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_size", "dataset_size", "val_loss", "dir"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
