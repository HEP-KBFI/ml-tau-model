import awkward as ak
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from mltau.tools import general as g

INPUT_PATH = (
    "/scratch/persistent/laurits/ml-tau/20260818_tauDaughterDataset/z_test.parquet"
)


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1


data = ak.from_parquet(INPUT_PATH)
vis_mass = g.reinitialize_p4(data.gen_jet_tau_p4).mass

decay_mode_name_mapping = {
    0: r"$h^{\pm}$",
    1: r"$h^{\pm}\pi^0$",
    2: r"$h^\pm+\geq2\pi^0$",
    10: r"$h^{\pm}h^{\mp}h^{\pm}$",
    11: r"$h^\pm h^\mp h^\pm$" "\n" r"$+\geq\pi^0$",
    15: "Rare",
}
bins = np.linspace(0, 1.78, num=25)

hep.style.use("CMS")
fig, ax = plt.subplots(figsize=(12, 8))
dm_vis_mass = []
labels = []
for dm, label in decay_mode_name_mapping.items():
    dm_mask = data.gen_jet_tau_decaymode == dm
    dm_vis_mass.append(vis_mass[dm_mask])
    labels.append(label)
ax.hist(
    dm_vis_mass,
    bins=bins,
    stacked=True,
    histtype="stepfilled",
    alpha=0.8,
    label=labels,
    density=True,
)

plt.legend()
plt.xlabel(r"$m_{vis}$ [GeV]")
plt.ylabel("Norm. count")
