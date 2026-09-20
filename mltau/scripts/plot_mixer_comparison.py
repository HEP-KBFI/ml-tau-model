import os
import importlib
import awkward as ak
import numpy as np
import torch
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from hydra import compose, initialize

from mltau.tools.evaluation import kinematics as k
from mltau.tools.evaluation import tagging as t
from mltau.tools.evaluation import charge_id as c
from mltau.tools.evaluation import decay_mode as d
from mltau.tools.general import reinitialize_p4

def main():
    # 1. Initialize config
    with initialize(version_base=None, config_path="mltau/config", job_name="comparison_plots"):
        cfg = compose(config_name="main")
    
    cfg.dataset.data_dir = "0528_Large_stats"
    
    RESULTS_DIR = "outputs/comparison_plots"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    SIGNAL_SAMPLE = "z"
    BKG_SAMPLE = "qq"
    CHARGE_FAKE_RATE_TARGET_EFFICIENCIES = [0.99, 0.95, 0.70]
    
    # Define directories for the three setups
    SETUP_DIRS = {
        "SingleParTau": { # Baseline ParT
            "is_tau": "outputs/0612_single_tauid_b8483f6",
            "decay_mode": "outputs/0612_single_decaymode_b8483f6",
            "charge": "outputs/0612_single_charge_b8483f6",
            "kinematics": "outputs/0612_single_kinematics_b8483f6",
        },
        "Mixer_standard": { # MLP-Mixer trained from scratch
            "is_tau": "outputs/0616_mixer_standard_is_tau_ca93231",
            "decay_mode": "outputs/0616_mixer_standard_decay_mode_ca93231",
            "charge": "outputs/0616_mixer_standard_charge_ca93231",
            "kinematics": "outputs/0616_mixer_standard_kinematics_ca93231",
        },
        "Mixer_distill": { # MLP-Mixer distilled from ParT
            "is_tau": "outputs/0616_mixer_distill_is_tau_ca93231",
            "decay_mode": "outputs/0616_mixer_distill_decay_mode_ca93231",
            "charge": "outputs/0616_mixer_distill_charge_ca93231",
            "kinematics": "outputs/0616_mixer_distill_kinematics_ca93231",
        }
    }
    
    # 2. Evaluate Tau ID (Tagging)
    print("Evaluating Tau ID (Tagging)...")
    tag_evaluators = []
    for algo, dirs in SETUP_DIRS.items():
        is_tau_dir = dirs["is_tau"]
        print(f"  Loading predictions for {algo} is_tau...")
        sig_path = os.path.join(is_tau_dir, "predictions", f"{SIGNAL_SAMPLE}_test.parquet")
        bkg_path = os.path.join(is_tau_dir, "predictions", f"{BKG_SAMPLE}_test.parquet")
        if not os.path.exists(sig_path) or not os.path.exists(bkg_path):
            print(f"  [WARNING] Predictions missing for {algo} is_tau. Skipping.")
            continue
        sig_data = ak.from_parquet(sig_path)
        bkg_data = ak.from_parquet(bkg_path)
        
        evaluator = t.TaggerEvaluator(
            signal_predictions=sig_data.tau_tagging_score,
            signal_gen_tau_p4=sig_data.gen_jet_tau_p4,
            signal_reco_jet_p4=sig_data.reco_jet_p4,
            bkg_predictions=bkg_data.tau_tagging_score,
            bkg_gen_jet_p4=bkg_data.gen_jet_p4,
            bkg_reco_jet_p4=bkg_data.reco_jet_p4,
            cfg=cfg,
            sample=SIGNAL_SAMPLE,
            algorithm=algo,
        )
        tag_evaluators.append(evaluator)
        
    if tag_evaluators:
        tRESULTS_DIR = os.path.join(RESULTS_DIR, "tau_id")
        os.makedirs(tRESULTS_DIR, exist_ok=True)
        tme = t.TaggerMultiEvaluator(tRESULTS_DIR, cfg)
        tme.combine_results(tag_evaluators)
        tme.save()
        print(f"Saved Tau ID plots to {tRESULTS_DIR}")
        
    # 3. Evaluate Decay Mode
    print("Evaluating Decay Mode...")
    dm_evaluators = []
    for algo, dirs in SETUP_DIRS.items():
        dm_dir = dirs["decay_mode"]
        print(f"  Loading predictions for {algo} decay_mode...")
        sig_path = os.path.join(dm_dir, "predictions", f"{SIGNAL_SAMPLE}_test.parquet")
        if not os.path.exists(sig_path):
            print(f"  [WARNING] Predictions missing for {algo} decay_mode. Skipping.")
            continue
        sig_data = ak.from_parquet(sig_path)
        
        evaluator = d.DecayModeEvaluator(
            pred_proba=sig_data.tau_decay_mode_probs,
            truth=sig_data.gen_jet_tau_decaymode,
            output_dir=os.path.join(RESULTS_DIR, "decay_mode"),
            sample=SIGNAL_SAMPLE,
            algorithm=algo,
        )
        dm_evaluators.append(evaluator)
        
    if dm_evaluators:
        dRESULTS_DIR = os.path.join(RESULTS_DIR, "decay_mode")
        os.makedirs(dRESULTS_DIR, exist_ok=True)
        dme = d.DecayModeMultiEvaluator(dRESULTS_DIR, cfg, sample=SIGNAL_SAMPLE)
        dme.combine_results(dm_evaluators)
        dme.save()
        print(f"Saved Decay Mode plots to {dRESULTS_DIR}")
        
    # 4. Evaluate Charge ID
    print("Evaluating Charge ID...")
    ch_evaluators = []
    for algo, dirs in SETUP_DIRS.items():
        ch_dir = dirs["charge"]
        print(f"  Loading predictions for {algo} charge...")
        sig_path = os.path.join(ch_dir, "predictions", f"{SIGNAL_SAMPLE}_test.parquet")
        if not os.path.exists(sig_path):
            print(f"  [WARNING] Predictions missing for {algo} charge. Skipping.")
            continue
        sig_data = ak.from_parquet(sig_path)
        
        evaluator = c.ChargeIdEvaluator(
            predicted=sig_data.tau_charge_score,
            truth=sig_data.gen_jet_tau_charge,
            gen_jet_tau_p4s=sig_data.gen_jet_tau_p4,
            reco_jet_p4s=sig_data.reco_jet_p4,
            cfg=cfg,
            output_dir=os.path.join(RESULTS_DIR, "charge_id"),
            sample=SIGNAL_SAMPLE,
            algorithm=algo,
        )
        ch_evaluators.append(evaluator)
        
    # Add QKappa reference using one of the available signal data
    # (since charge prediction is signal-only) Load Mixer_distill predictions to get correct shape
    sig_path = os.path.join(SETUP_DIRS["Mixer_distill"]["charge"], "predictions", f"{SIGNAL_SAMPLE}_test.parquet")
    qk_sample = ak.from_parquet(sig_path) if os.path.exists(sig_path) else None
            
    if qk_sample is not None:
        print("  Creating QKappa reference...")
        def calculate_qkappa(cand_p4, jet_p4, cand_charges, best_kappa=0.5):
            cand_pts = reinitialize_p4(cand_p4).pt
            jet_pts = reinitialize_p4(jet_p4).pt
            numerator = np.sum(cand_charges * cand_pts ** best_kappa, axis=1)
            denominator = jet_pts ** best_kappa
            qkappa_charge = numerator / np.where(denominator > 0, denominator, 1.0)
            qkappa_charge_score = np.clip(0.5 * (qkappa_charge + 1.0), 0.0, 1.0)
            return qkappa_charge_score
            
        cand_p4 = reinitialize_p4(qk_sample.cand_p4)
        jet_p4 = reinitialize_p4(qk_sample.reco_jet_p4)
        cand_charges = qk_sample.cand_charges
        
        qkappa_charge_score = calculate_qkappa(cand_p4, jet_p4, cand_charges)
        
        qkCh_evaluator = c.ChargeIdEvaluator(
            predicted=qkappa_charge_score,
            truth=qk_sample.gen_jet_tau_charge,
            gen_jet_tau_p4s=qk_sample.gen_jet_tau_p4,
            reco_jet_p4s=qk_sample.reco_jet_p4,
            cfg=cfg,
            output_dir=os.path.join(RESULTS_DIR, "charge_id"),
            sample=SIGNAL_SAMPLE,
            algorithm="QKappa",
        )
        ch_evaluators.append(qkCh_evaluator)
        
    if ch_evaluators:
        cRESULTS_DIR = os.path.join(RESULTS_DIR, "charge_id")
        os.makedirs(cRESULTS_DIR, exist_ok=True)
        cme = c.ChargeMultiEvaluator(
            cRESULTS_DIR,
            cfg,
            target_efficiencies=CHARGE_FAKE_RATE_TARGET_EFFICIENCIES,
        )
        cme.combine_results(ch_evaluators)
        cme.save()
        print(f"Saved Charge ID plots to {cRESULTS_DIR}")
        
    # 5. Evaluate Kinematics
    print("Evaluating Kinematics...")
    kin_evaluators = []
    for algo, dirs in SETUP_DIRS.items():
        kin_dir = dirs["kinematics"]
        print(f"  Loading predictions for {algo} kinematics...")
        sig_path = os.path.join(kin_dir, "predictions", f"{SIGNAL_SAMPLE}_test.parquet")
        if not os.path.exists(sig_path):
            print(f"  [WARNING] Predictions missing for {algo} kinematics. Skipping.")
            continue
        sig_data = ak.from_parquet(sig_path)
        
        evaluator = k.KinematicsEvaluator(
            predicted_p4=sig_data.tau_p4,
            true_p4=sig_data.gen_jet_tau_p4,
            cfg=cfg,
            algorithm=algo,
            sample_name=SIGNAL_SAMPLE,
        )
        kin_evaluators.append(evaluator)
        
    # Add RecoJet reference kinematics
    if kin_evaluators:
        print("  Creating RecoJet reference...")
        rKin_evaluator = k.KinematicsEvaluator(
            predicted_p4=sig_data.reco_jet_p4,
            true_p4=sig_data.gen_jet_tau_p4,
            cfg=cfg,
            algorithm="RecoJet",
            sample_name=SIGNAL_SAMPLE,
        )
        kin_evaluators.append(rKin_evaluator)
        
        kRESULTS_DIR = os.path.join(RESULTS_DIR, "kinematics")
        os.makedirs(kRESULTS_DIR, exist_ok=True)
        kme = k.KinematicsMultiEvaluator(kRESULTS_DIR, cfg, sample=SIGNAL_SAMPLE)
        kme.combine_results(kin_evaluators)
        kme.save()
        print(f"Saved Kinematics plots to {kRESULTS_DIR}")
        
    print("\nAll evaluations complete!")

if __name__ == "__main__":
    main()
