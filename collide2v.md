---
# Dataset Card for Hugging Face Hub

pretty_name: COLLIDE-2V (Dual-View HL-LHC Simulation Corpus)
dataset_name: collide-2v
dataset_summary: >
  COLLIDE-2V is a large-scale, dual-view (trigger-level and offline) simulated dataset
  for high-energy physics at HL-LHC conditions (μ≈200 pileup). Each event is produced with
  a MG5_aMC → PYTHIA8 → Delphes pipeline and written with aligned views and multi-tier
  representations (generator/particle/detector). The corpus targets foundation-model
  pretraining, transfer learning, and benchmarking across reconstruction, triggering,
  anomaly detection, and generative modeling.
task_categories:
  - other
  - domain-adaptation
  - regression
  - classification
  - anomaly-detection
  - generative-modeling
  - multi-view-learning
languages:
  - no-language
licenses:
  - cc-by-4.0  # or another license; update before release
multilinguality:
  - not-applicable
size_categories:
  - 100M<n<1B
source_datasets:
  - original
annotations_creators:
  - no-annotation
language_creators:
  - no-annotation
paperswithcode_id:  # optional; fill if a paper is linked
pretty_urls:
  paper:  # add arXiv or DOI after submission
  code:   # add loaders/baselines repo
---

# COLLIDE-2V: Dual-View HL-LHC Dataset

> **Status:** Public preview. Replace all `TBD` placeholders prior to release.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Supported Tasks and Benchmarks](#supported-tasks-and-benchmarks)
- [How to Use](#how-to-use)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
  - [File Layout and Sharding](#file-layout-and-sharding)
  - [Precision and Units](#precision-and-units)
- [Data Generation and Processing](#data-generation-and-processing)
  - [Simulation Chain](#simulation-chain)
  - [Dual-View Alignment](#dual-view-alignment)
  - [Post-processing](#post-processing)
  - [Quality Control](#quality-control)
- [Performance Tips](#performance-tips)
- [Limitations, Biases, and Ethical Considerations](#limitations-biases-and-ethical-considerations)
- [Dataset Governance and Versioning](#dataset-governance-and-versioning)
- [Licensing Information](#licensing-information)
- [Citation](#citation)
- [Contributions](#contributions)
- [Changelog](#changelog)
- [FAQ](#faq)

---

## Dataset Description

**Summary.** COLLIDE-2V (“dual-view”) is a near-billion-event HL-LHC simulation corpus designed for AI-native physics research. Each event is represented at three tiers:

- **Generator level (truth)**: parton/particle kinematics and associations.
- **Reconstruction level (offline view)**: full PF/PUPPI reconstruction typical of analysis workflows.
- **Trigger level (L1T view)**: low-latency, coarser-granularity objects suitable for online selection.

Events in the two detector views are **aligned one-to-one**, enabling contrastive/self-supervised learning, distillation (offline → trigger), and systematic studies of domain shift.

**Motivation.** The HL-LHC era introduces extreme pileup and throughput constraints. ML methods—particularly foundation models—benefit from large, diverse, **multi-view** corpora. COLLIDE-2V provides such a corpus with a unified schema and standardized metadata to promote reproducibility and community baselines.

**Use cases (non-exhaustive).**

- Representation learning and pretraining (contrastive, masked modeling, autoencoding).
- Trigger emulation and optimization (turn-on curves at fixed rates).
- Object/event reconstruction, calibration, and uncertainty modeling.
- Anomaly detection and new-physics searches (C2ST, iFAR metrics).
- Conditional and physics-aware generative modeling.
- Domain adaptation between L1T and offline views.

---

## How to Use

### Quick start (streaming)
#### EOS
For using the dataset with EOS access on the CERN Lxplus batch system, the following repository is provided: [foundation_model_testing](https://github.com/pploner/foundation_model_testing/tree/main).

The repository provides a Pytorch Lightning based dataloader, preprocessing functionality and toy MLP and transformer models that can be configured via a Hydra config. It also supports batch submission with the HTCondor system on Lxplus for CPU and GPU based runs, and in particular parallel multi-node distributed batch submission to speed up vectorization and preprocessing.

We also provide sample notebooks and plotting scripts for dataset inspection, as well as compatibility with the Mlflow logger and hyperparameter sweeping with Optuna.

More detailled documentation can be found in the corresponding readme file and the various .yaml Hydra config files in `configs/`.

#### Hugging Face
We are planning to host the dataset on Hugging Face. A Hugging Face streaming compatible version of the afforementioned testing repository will be produced to also make it easy get started with this version of the dataset.

# Dataset Structure

## Data Instances


---

## Data Fields

Scalar kinematic quantities (`PT`, `ET`, `Eta`, `Phi`, `Mass`, `E`, `MET`) are stored as **float16** on disk and may be decoded to **float32** in memory by the loader (configurable).

### Common top-level fields
- event_id *(string/int64)* — unique event key.  
- process *(string)* — generator-level process label.  

### Reconstructed views

Both reconstructed views share the same **physics objects** but differ in reconstruction/selection to emulate:
- **Offline / FullReco**: standard CMS-style offline reconstruction - Particle Flow (PF).
- **L1T**: ultra-fast trigger-style reconstruction.

All object collections are stored as **ragged** per-event arrays (`dict of lists`). Collections that carry a `Constituents` field use **view-local references** to candidate `fUniqueID`s.

### Offline / FullReco view

##### PUPPI and PF candidates
- **`PUPPIPart` (EFlowPuppi)** *(dict of lists)*:  
  `PT, Eta, Phi, E, Charge, Mass, PID, D0, DZ, ErrorD0, ErrorDZ, fUniqueID, PuppiW, IsPU, IsRecoPU`
- **`PFPart` (EFlow)** *(dict of lists)*:  
  `PT, Eta, Phi, E, Charge, Mass, PID, D0, DZ, ErrorD0, ErrorDZ, fUniqueID, PuppiW, IsPU, IsRecoPU`

##### High-level leptons and photons
- **`Electron` (Electron)**:  
  `PT, Eta, Phi, Charge, EhadOverEem, IsolationVar, IsolationVarRhoCorr, D0, DZ, ErrorD0, ErrorDZ`
- **`MuonTight` (MuonTight)**:  
  `PT, Eta, Phi, Charge, IsolationVar, IsolationVarRhoCorr, D0, DZ, ErrorD0, ErrorDZ`
- **`PhotonTight` (PhotonTight)**:  
  `PT, Eta, Phi, EhadOverEem, IsolationVar, IsolationVarRhoCorr`

##### Jets
> **`Constituent pointer conventions (per event, view-local):`**
`Constituents`: list of candidate **IDs** matching the target collection’s `fUniqueID`.
`ConstituentIdx`: list of candidate **indices** into the target collection (0-based)

>**`Flavor` note (MC truth):** `Jet*.Flavor` is a generator-derived jet flavor label (truth-matched annotation) provided for supervision/validation.  
It should generally **not** be used as an input feature for physics-realistic inference unless you are explicitly studying label leakage or training with privileged information.

- **`JetAK4` (Jet)** *(PF constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `PFPart.fUniqueID`\
  *ConstituentsIdx →* `PFPart[i]`

- **`JetAK8` (JetAK8)** *(PF constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `PFPart.fUniqueID`\
  *ConstituentsIdx →* `PFPart[i]`

- **`JetPuppiAK4` (JetPUPPI)** *(PUPPI constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `PUPPIPart.fUniqueID`\
  *ConstituentsIdx →* `PUPPIPart[i]`

- **`JetPuppiAK8` (JetPUPPIAK8)** *(PUPPI constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `PUPPIPart.fUniqueID`\
  *ConstituentsIdx →* `PUPPIPart[i]`

##### Pileup density and scalar sums
- **`Rho` (Rho)**: `Rho`
- **`ScalarHT` (ScalarHT)**: `HT`

##### Missing transverse momentum (MET)
- **`MET` (MissingET)**: `MET, Eta, Phi`
- **`PUPPIMET` (PuppiMissingET)**: `MET, Eta, Phi`

### L1T view
#### Recommended L1T usage

The L1T view is designed to emulate the dominant practical constraint of real-time trigger reconstruction: **a hard cap on the number of particle-flow candidates processed per event**. However, the limit is not explicitly enforced during generation. 

#### Canonical “hardware-like” candidate cap

For the most realistic L1T benchmark setting, we recommend enforcing a **fixed per-event limit of 128 candidates**, keeping the **highest-`pT`** candidates only:

- **Target cap:** `N = 128`
- **Selection rule:** As particle collections already arrive sorted by descending `PT` keep the first `N = 128` particles.
- **Scope:** apply to the L1T candidate collections (e.g. `L1TPFPart`, `L1TPUPPIPart`) before forming model inputs.

This setting matches the typical technical limitation in low-latency trigger pipelines (bounded bandwidth / fixed compute), and it makes model performance comparisons more meaningful and reproducible.


> Note: the truncation is an **evaluation-time / input-pipeline choice** (not a requirement of the dataset itself). The dataset stores the full available L1T collections so users can study the impact of different resource budgets.

The L1T view uses the **same field lists** as Offline / FullReco, but collection names are prefixed with `L1T`:

##### PUPPI and PF candidates
- **`L1TPUPPIPart`**:  
  `PT, Eta, Phi, E, Charge, Mass, PID, D0, DZ, ErrorD0, ErrorDZ, fUniqueID, PuppiW, IsPU, IsRecoPU`
- **`L1TPFPart`**:  
  `PT, Eta, Phi, E, Charge, Mass, PID, D0, DZ, ErrorD0, ErrorDZ, fUniqueID, PuppiW, IsPU, IsRecoPU`

##### High-level leptons and photons
- **`L1TElectron`**:  
  `PT, Eta, Phi, Charge, EhadOverEem, IsolationVar, IsolationVarRhoCorr, D0, DZ, ErrorD0, ErrorDZ`
- **`L1TMuonTight`**:  
  `PT, Eta, Phi, Charge, IsolationVar, IsolationVarRhoCorr, D0, DZ, ErrorD0, ErrorDZ`
- **`L1TPhotonTight`**:  
  `PT, Eta, Phi, EhadOverEem, IsolationVar, IsolationVarRhoCorr`

##### Jets

>**`Constituent pointer conventions (per event, view-local):`**
`Constituents`: list of candidate **IDs** matching the target collection’s `fUniqueID`.
`ConstituentIdx`: list of candidate **indices** into the target collection (0-based)

>**`Flavor` note (MC truth):** `L1TJet*.Flavor` is a generator-derived jet flavor label (truth-matched annotation) provided for supervision/validation.  
It should generally **not** be used as an input feature for physics-realistic inference unless you are explicitly studying label leakage or training with privileged information.

- **`L1TJetAK4`** *(PF constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `L1TPFPart.fUniqueID`\
  *ConstituentsIdx →* `L1TPFPart[i]`

- **`L1TJetAK8`** *(PF constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `L1TPFPart.fUniqueID`\
  *ConstituentsIdx →* `L1TPFPart[i]`

- **`L1TJetPuppiAK4`** *(PUPPI constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `L1TPUPPIPart.fUniqueID`\
  *ConstituentsIdx →* `L1TPUPPIPart[i]`
- **`L1TJetPuppiAK8`** *(PUPPI constituents)*:  
  `PT, Eta, Phi, Mass, BTag, BTagPhys, NCharged, NNeutrals, Charge, Flavor, Constituents`  
  *Constituents →* `L1TPUPPIPart.fUniqueID`\
  *ConstituentsIdx →* `L1TPUPPIPart[i]`
##### Pileup density and scalar sums
- **`L1TRho`**: `Rho`
- **`L1TScalarHT`**: `HT`

##### Missing transverse momentum (MET)
- **`L1TMET`**: `MET, Eta, Phi`
- **`L1TPUPPIMET`**: `MET, Eta, Phi`

### Generator level (truth)

Truth collections are provided independent of view.

- **`GenPart` (Particle)** *(dict of lists)*:  
  `PT, Eta, Phi, Mass, PID, M1, M2, D1, D2, Status, IsPU`
- **`GenJetAK4` (GenJet)**: `PT, Eta, Phi, Mass`
- **`GenJetAK8` (GenJetAK8)**: `PT, Eta, Phi, Mass`
- **`GenMissingET` (GenMissingET)**: `MET, Eta, Phi`
- Some truth-matched annotations (e.g. `*Jet*.Flavor`) may appear on reconstructed objects for convenience; treat them as **gen labels**, not reco-only features.
### Event-level physics and metadata


#### Vertices
- **`Vertex`**:  
  `X, Y, Z, T, Index, NDF, SumPT2, Constituents`

#### Event Info (Bookkeeping)

- **`EventInfo` (Event)** *(dict of scalars)*:  
  `Number, ProcessID, Weight, CrossSection, CrossSectionError, Scale, AlphaQED, AlphaQCD, ID1, ID2, X1, X2, ScalePDF, PDF1, PDF2`

> See the **Schema JSON** in each release for precise `dtype` and nullability.

---

## Data Splits
- `train` — default set (majority of shards).  
- `validation` — fixed subset; do not use for model selection leakage across tasks.  
- `test` — held-out; keep for final reporting.

**Split strategy:** deterministic shard-level hashing by `event_id` to avoid leakage.

---

## File Layout and Sharding
- **Storage format:** Parquet. (root deleted at generation)
- **Shards:** single-process, each ~10,000 events (set to meet I/O goals).  
- **Naming convention:** `process={NAME}/split={train|validation|test}/part-{00000}.parquet`.  
- **Metadata files:**
  - `DATA_CARD.json` — software versions, card SHAs, PU, seeds.
  - `SCHEMA.json` — column names, dtypes, list offset encodings.
  - `CHECKSUMS.txt` — SHA256 for every file.

---

## Precision and Units
- **float16** for `PT, ET, Eta, Phi, Mass, E, MET` (on-disk), validated for dynamic range; overflow values are clamped and flagged via `__cast_ok` bitmasks in the schema (optional).
- Other floats: **float32/float64** as appropriate.
- Units: **GeV** for momenta/energies, **radians** for angles, **cm** for positions, **ns** for times, charge in units of **|e|**.

---

# Data Generation and Processing

## Simulation Chain
1. **Matrix-element generation (MG5\_aMC).**  
   - Tool: *MadGraph5\_aMC@NLO*, version 3.5.8.  
   - Assets archived: process and run cards, UFO models, random seeds, nominal cross sections, PDF settings.
2. **Parton shower & hadronization (PYTHIA8).**  
   - Tool: *PYTHIA8*, version 8.310.  
   - Tunes and merging parameters recorded per process.  
   - **Pileup overlay:** Poisson(μ≈200) using inelastic pp minimum-bias; per-file PU seed stored.
3. **Fast detector simulation (Delphes).**  
   - Tool: *Delphes* 3.5.0 with two cards:
     - **Offline view:** CMS Phase-2 HL-LHC card (PF + PUPPI, small-R/large-R jets, e/μ/γ selections).
     - **L1T view:** CMS Phase-2 L1T TDR-inspired card (reduced resolution/latency).

> **Simulation / reconstruction configuration:** COLLIDE-2V samples are produced with the official Delphes Phase-II configuration card  > **`CMS_PhaseII_200PU_COLLIDE2V_v1.tcl`** (modules, object definitions, and collection names):  
> [fermilab-hep-ai/simulation — CMS_PhaseII_200PU_COLLIDE2V_v1.tcl](https://github.com/fermilab-hep-ai/simulation/blob/main/cards/CMS_PhaseII_200PU_COLLIDE2V_v1.tcl)
   - Card SHAs and any edits are versioned in the release.

## Dual-View Alignment
- Both detector views originate from the same particle-level event.  
- `event_id` uniquely identifies the event in all tiers.  
- Constituent references (e.g., `Constituents`) are view-local and resolve to `fUniqueID` in the corresponding PF/PUPPI collection.

## Post-processing
- **Export:** ROOT → Parquet (ZSTD, block ~128 KB). Optional HDF5 mirrors.  
- **Normalization:** enforce units, angle ranges (φ ∈ (−π, π]), charge units.  
- **Precision casting:** float16 casting with range checks and optional `__cast_ok` flags.  
- **Manifests:** per-release `DATA_CARD.json`, `SCHEMA.json`, `CHECKSUMS.txt`.

## Quality Control
- **Schema validation:** field names, dtypes, list offsets.  
- **Physics sanity:** multiplicities, kinematic ranges, MET consistency, jet↔constituent integrity.  
- **Determinism:** small # Dataset Structure


---

# Performance Tips
- Prefer **streaming** (`streaming=True`) for rapid prototyping and cluster training.  
- Use `.with_format("torch")` / `.with_format("numpy")` to avoid Python object conversion.  
- For ragged lists, predefine **padding/truncation** lengths per task; cache with `.map(..., batched=True)` to Arrow.  
- When training across views, use the **dual** config to avoid manual joins.  
- Pin threads for Arrow/Parquet (`OMP_NUM_THREADS`, `ARROW_NUM_THREADS`) to match your I/O topology.

---

# Limitations, Biases, and Ethical Considerations
- **Simulation-specific artifacts.** Detector and reconstruction reflect particular Delphes cards; real-data calibrations and time-dependent effects are not present by default.  
- **Process coverage.** Despite broad coverage, extremely rare corners and some detector aging scenarios are not included in v1.0.  
- **Generalization.** Models pretrained here should be validated on real open data where appropriate; include uncertainty estimates.  
- **Ethics & safety.** This dataset represents **simulated** particle-physics events. It contains no personal data. Do not use for safety-critical decisions without domain validation.

---

# Dataset Governance and Versioning
- **Versioning:** Semantic versions (`v1.0.0`, `v1.1.0`, …) with DOIs per major release (TBD).  
- **Provenance:** All software version strings, cards, seeds, and SHAs are shipped in `DATA_CARD.json`.  
- **Reproducibility:** Re-running with provided seeds/cards reproduces identical shards.  
- **Issue tracking:** Use the GitHub issue tracker in the companion repository (TBD).  
- **Community contributions:** See **Contributions**.

---

# Licensing Information
- **Dataset license:** `CC-BY-4.0` (TBD; confirm before release).  
- **Third-party tools:** Respect licenses for *MadGraph5\_aMC@NLO*, *PYTHIA8*, *Delphes*, and any UFO models or tunes used. Cite accordingly.seed range replays bit-identical outputs.  
- **Spot validations:** mass peaks, response/resolution vs pT/η (plots shipped in `validation/`).

---
