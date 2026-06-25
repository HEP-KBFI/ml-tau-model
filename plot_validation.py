import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def plot_validation(input_path, output_path):
    print(f"Reading {input_path} ...")
    df = ak.from_parquet(input_path)
    
    # Filter for matched taus
    mask = df["is_tau"] == 1
    if np.sum(mask) == 0:
        print("No matched taus found in the dataset.")
        return

    # Extract momenta
    # gen_jet_tau_p4s: [pt, eta, phi, mass]
    # reco_jet_p4s: [pt, eta, phi, mass]
    gen_pt = np.array([p[0] for p in df["gen_jet_tau_p4s"][mask]])
    reco_pt = np.array([p[0] for p in df["reco_jet_p4s"][mask]])

    # Setup plot style
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(gen_pt, reco_pt, alpha=0.6, label=f"Matched Jets (N={len(gen_pt)})")
    
    # Identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Identity")
    
    ax.set_xlabel(r"Gen Tau Visible $p_T$ [GeV]")
    ax.set_ylabel(r"Matched Jet $p_T$ [GeV]")
    ax.set_title("Gen vs Reco Momentum Validation")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_validation("ggHtautau_jet_level.parquet", "gen_vs_reco_pt.png")
