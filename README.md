# End-to-End ML Reconstruction and Identification of Hadronically Decaying Tau Leptons

The aim of this project is to develop and test end-to-end machine learning methods for reconstruction and identification of hadronically decaying tau leptons, while also providing a thoroughly validated and tested dataset for evaluating the performance of said algorithms.

This repository contains the **ParTau** model — a multi-task [Particle Transformer (ParT)](https://arxiv.org/abs/2202.03772) adapted for future lepton collider experiments (CLD/CLIC/FCC-ee). Given a reconstructed jet and its constituent particles, ParTau simultaneously predicts:

- **Tau tagging** — whether the jet originates from a hadronic tau decay
- **Decay mode classification** — 6-class HPS-aligned decay mode  
- **Visible tau kinematics** — $p_T$ ratio, direction correction ($\Delta\theta$, $\Delta\phi$), and visible mass via regression
- **Tau charge** — binary (+1 or −1) classification

Tau leptons can decay both leptonically and hadronically, however only hadronic decays are targeted with this project.

## Publications

The results of these studies have been divided across three separate papers, covering tau identification, kinematic and decay mode reconstruction, and foundation model approaches.

### Tau lepton identification and reconstruction: a new frontier for jet-tagging ML algorithms [![DOI:10.1016/j.cpc.2024.109095](http://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109095-f9f107.svg)](https://doi.org/10.1016/j.cpc.2024.109095) [![arXiv](https://img.shields.io/badge/arXiv-2307.07747-b31b1b.svg)](https://arxiv.org/abs/2307.07747)

_[Published in: Comput.Phys.Commun. 298 (2024) 109095]_

**Keywords:** tau tagging

We systematically compare state-of-the-art deep learning architectures for hadronically decaying tau lepton identification, including ParticleTransformer, LorentzNet, and HPS + DeepTau variants. The study demonstrates that jet-tagging ML algorithms can be successfully adapted for tau identification tasks, with ParticleTransformer achieving optimal performance for future lepton collider experiments.

---

### A unified machine learning approach for reconstructing hadronically decaying tau leptons [![DOI:10.1016/j.cpc.2024.109399](http://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109399-f9f107.svg)](https://doi.org/10.1016/j.cpc.2024.109399) [![arXiv](https://img.shields.io/badge/arXiv-2407.06788-b31b1b.svg)](https://arxiv.org/abs/2407.06788)

_[Published in: Comput.Phys.Commun. 307 (2025) 109399]_

**Keywords:** decay mode classification, kinematic regression

We present a multi-task learning framework that decomposes tau lepton reconstruction into two sub-tasks: kinematic reconstruction, and decay mode classification. The study compares state-of-the-art architectures, achieving momentum resolutions of 2-3% and decay mode classification precision between 80-95%, with ParticleTransformer significantly outperforming heuristic baselines.

---

### Reconstructing hadronically decaying tau leptons with a jet foundation model [![DOI:10.21468/SciPostPhysCore.8.3.046](http://img.shields.io/badge/DOI-10.21468/SciPostPhysCore.8.3.046-f9f107.svg)](https://doi.org/10.21468/SciPostPhysCore.8.3.046) [![arXiv](https://img.shields.io/badge/arXiv-2503.19165-b31b1b.svg)](https://arxiv.org/abs/2503.19165)

_[Published in: SciPost Phys. Core 8, 046 (2025)]_

**Keywords:** foundation models, transfer learning, tau tagging, decay mode classification, kinematic regression

We study how OmniJet-α, a foundation model for particle jets, can be fine-tuned for hadronically decaying tau lepton reconstruction, demonstrating ~50% improvement in momentum reconstruction resolution when using pretrained weights compared to training from scratch.

---

## Dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13881061.svg)](https://doi.org/10.5281/zenodo.13881061)

The dataset contains 2 signal samples (ZH→Ztautau and Z→tautau) and one background sample (Z→qq), based on CLD detector simulation using the key4hep framework for $e^+e^-$ collision events.

## Quick Start

### Local Development (Custom Environment)
For local training on your machine without requiring the exact same environment as the authors:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 mltau/scripts/train.py
```

### HPC Environment (Frozen Singularity Image)
For training in HPC environments with the frozen singularity image containing all required dependencies:

```bash
# Local training with containerized environment
./run.sh python3 mltau/scripts/train.py

# Cluster training (SLURM)
sbatch train-gpu.sh

# Custom configuration
./run.sh python3 mltau/scripts/train.py training.lr=5e-4 training.max_epochs=200
```

### Running Inference
To generate prediction parquet files from a pretrained model checkpoint:

```bash
./run_inference.sh <path_to_checkpoint> <output_directory>
```
This will run inference on the test dataset and save `.parquet` files to `<output_directory>/predictions/`.

### Evaluation & Plotting
Final physics performance plots (ROC, efficiency, resolution) are typically generated via Jupyter notebooks:
1. Open `mltau/notebooks/MultiParTau_vs_SingleParTau.ipynb`.
2. Update the `OUTPUT_DIR` and `SIGNAL_SAMPLE`/`BKG_SAMPLE` paths to point to your inference outputs.
3. Run the cells to generate the PDF plots.

## Documentation

- **[Technical Details](docs/TECHNICAL.md)**: Complete model architecture, training setup, and implementation details
- **Configuration**: See [mltau/config/](mltau/config/) for all configurable parameters
- **Examples**: Training scripts and usage examples in [mltau/scripts/](mltau/scripts/)

## Project Structure

```
mltau/
├── config/           # Hydra configuration files  
├── models/           # ParTau and ParticleTransformer implementations
├── scripts/          # Training and inference scripts
└── tools/            # Feature extraction, evaluation, and logging utilities
docs/                 # Technical documentation
```

The ParTau model successfully demonstrates state-of-the-art performance for multi-task tau reconstruction, providing a robust foundation for future lepton collider experiments.