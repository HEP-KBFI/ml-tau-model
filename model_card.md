---
tags:
- particle-physics
- tau-reconstruction
- fcc-ee
- key4hep
- particle-transformer
---

# ML-Tau Model Card

This repository contains models for tau reconstruction and identification at future colliders (FCC), based on the Particle Transformer (ParT) architecture.

## Dataset
- **Name:** `0509_dsinphi_to_sindphi`
- **Source:** Preprocessed jet-based FCC dataset for hadronic tau reconstruction.
- **Physics Processes:**
  - **Signal:** $Z \to \tau^+\tau^-$ events.
  - **Background:** $Z \to q\bar{q}$ (light quarks and gluons).
- **Generation & Simulation:**
  - **Generator:** Pythia8
  - **Detector Model:** CLD (`CLD_o2_v07`) for FCC-ee.
  - **Simulation:** Geant4 (via `ddsim`).
  - **Software Stack:** [Key4hep Project](https://github.com/key4hep) (release 2025-05-29). [Key4hep-sim (v1.2.5)](https://github.com/HEP-KBFI/key4hep-sim/tree/v1.2.5)
  - **Reconstruction:** Standard CLD reconstruction (`CLDReconstruction.py`).
- **Split:** 70% Training, 10% Validation, 20% Testing.
- **Input Features:** 17 candidate-level features (kinematics, identification, etc.).
- **Jet Composition:** Maximum of 20 candidates per jet.

## Dataset Statistics
- **Total Jets:** 9,049,163
- **Signal (Tau) Jets:** 1,187,870
- **Background (Quark/Gluon) Jets:** 7,861,293
- **Training Set:** 7,075,163 background + 1,069,083 signal jets
- **Test Set:** 786,130 background + 118,787 signal jets

## Model Architecture
The models utilize the **Particle Transformer (ParT)** architecture, which uses a combination of particle-level and pair-level features to learn jet representations.

### Variants
- **MultiParTau:** A multi-task learning model that simultaneously performs four tasks:
  - **Tau Identification (`is_tau`):** Binary classification (Signal tau vs. Quark/Gluon jet).
  - **Charge Classification:** Identification of the tau charge (+1 or -1).
  - **Decay Mode Classification:** 6-class classification of tau decay modes.
  - **Kinematics Regression:** Prediction of 5 kinematic corrections: `[log(pt_gen/pt_reco), delta_eta, delta_sin(phi), delta_cos(phi), log(m_gen/m_reco)]`.
- **SingleParTau:** Specialized models trained for one of the above tasks individually.

### Hyperparameters
- **Embedding Dimensions:** `[256, 512, 256]`
- **Pair Embedding Dimensions:** `[64, 64, 64]`
- **Attention Heads:** 8
- **Transformer Layers:** 8 (default) / 2 (specific runs)
- **CLS Layers:** 2
- **Activation:** GELU

## Training Scheme
- **Optimizer:** AdamW with a weight decay of 1e-2.
- **Learning Rate:** 0.001 (base).
- **Scheduler:** `OneCycleLR` with cosine annealing.
- **Batch Size:** 12288.
- **Precision:** 16-mixed (FP16).
- **Multi-task Strategy (MultiParTau):** PCGrad (Projected Conflicting Gradients) is employed to handle gradient conflicts between different tasks during training.
- **Task Weighting:**
  - Tau ID: 1.0
  - Charge: 1.0
  - Decay Mode: 1.0
  - Kinematics: 2.0

## Trained Models & Git Hashes
The models located in the `outputs/0609_*` directories correspond to the following configurations and git hashes:

| Model Name | Task | Git Hash |
|------------|------|----------|
| `multipartau_full` | Multi-task | `fef7149` |
| `single_charge` | Charge | `a5d7ed8` |
| `single_decaymode` | Decay Mode | `a5d7ed8` |
| `single_kinematics` | Kinematics | `a5d7ed8` |
| `single_tauid` | Tau ID | `a5d7ed8` |

---
*Generated based on training configurations and experiment outputs.*
