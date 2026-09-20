import csv
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot scaling study results.")
    parser.add_argument("input_csv", help="CSV file with results (e.g. scaling_results.csv)")
    parser.add_argument("--output", default="scaling_study.png", help="Output plot file")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: File {args.input_csv} does not exist.")
        return

    results = []
    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["dataset_size"] = int(row["dataset_size"])
            row["val_loss"] = float(row["val_loss"])
            results.append(row)

    model_sizes = sorted(list(set(r["model_size"] for r in results)))
    
    plt.figure(figsize=(10, 6))
    
    for m_size in model_sizes:
        m_results = [r for r in results if r["model_size"] == m_size]
        m_results.sort(key=lambda x: x["dataset_size"])
        
        ds_sizes = [r["dataset_size"] for r in m_results]
        losses = [r["val_loss"] for r in m_results]
        
        plt.plot(ds_sizes, losses, marker='o', label=f"Model: {m_size}")

    plt.xlabel("Dataset Size")
    plt.ylabel("Validation Loss")
    plt.title("Scaling Study: Validation Loss vs Dataset Size")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale("log")
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
