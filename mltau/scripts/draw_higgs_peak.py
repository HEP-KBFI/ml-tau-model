#!/usr/bin/env python3
import sys
import os
import glob
import argparse
import torch
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import vector
import uproot
import fastjet
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize

# Add relevant paths to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "ml-tau-data"))

# Project imports
from mltau.models import MultiParTau_module
from mltau.tools.evaluation.inference import decode_kinematic_predictions
from mltau.tools.general import reinitialize_p4
from ntupelizer.tools import clustering as cl
from ntupelizer.tools import particle_filters as pfl
from ntupelizer.tools import lifetime as lt
from ntupelizer.tools import general as g
from ntupelizer.tools import features as f
from ntupelizer.tools.tau_decaymode import get_reduced_decaymodes

import multiprocessing as mp

# Constants
MAX_CANDS = 20
TAU_ID_THRESHOLD = 0.9

def get_branches():
    return [
        "MCParticles.PDG", "MCParticles.generatorStatus", "MCParticles.momentum.x", "MCParticles.momentum.y", "MCParticles.momentum.z", "MCParticles.mass",
        "MCParticles.parents_begin", "MCParticles.parents_end", "MCParticles.daughters_begin", "MCParticles.daughters_end",
        "MCParticles.endpoint.x", "MCParticles.endpoint.y", "MCParticles.endpoint.z",
        "MCParticles.vertex.x", "MCParticles.vertex.y", "MCParticles.vertex.z",
        "_MCParticles_parents.index", "_MCParticles_daughters.index",
        "PandoraPFOs.PDG", "PandoraPFOs.energy", "PandoraPFOs.momentum.x", "PandoraPFOs.momentum.y", "PandoraPFOs.momentum.z", "PandoraPFOs.charge", "PandoraPFOs.mass",
        "PandoraPFOs.tracks_begin", "PandoraPFOs.tracks_end",
        "_SiTracks_Refitted_trackStates.D0", "_SiTracks_Refitted_trackStates.Z0", "_SiTracks_Refitted_trackStates.phi", "_SiTracks_Refitted_trackStates.tanLambda", "_SiTracks_Refitted_trackStates.omega",
        "_SiTracks_Refitted_trackStates.referencePoint.x", "_SiTracks_Refitted_trackStates.referencePoint.y", "_SiTracks_Refitted_trackStates.referencePoint.z",
        "_SiTracks_Refitted_trackStates.covMatrix.values[21]", "_SiTracks_Refitted_trackStates.location",
        "PrimaryVertices.position.x", "PrimaryVertices.position.y", "PrimaryVertices.position.z",
        "_RecoMCTruthLink_from.index", "_RecoMCTruthLink_to.index", "RecoMCTruthLink.weight",
        "_PandoraPFOs_tracks.index", "_PandoraPFOs_tracks.collectionID"
    ]

from ntupelizer.tools import matching as m
from ntupelizer.tools import gen_tau_info_matcher as gtim

def process_single_file(args):
    file_path, ckpt_path, cfg, device_str, max_events, is_signal = args
    device = torch.device(device_str)
    
    # Load model in worker
    from omegaconf.dictconfig import DictConfig as OmegaDictConfig
    torch.serialization.add_safe_globals([OmegaDictConfig])
    model = MultiParTau_module.ParTauModule.load_from_checkpoint(
        ckpt_path, cfg=cfg, input_dim=17, num_dm_classes=6, weights_only=False
    )
    model.to(device)
    model.eval()
    
    file_masses = []
    file_gen_masses = []
    file_jet_pts = []
    branches = get_branches()
    
    print(f"Processing {file_path}...")
    with uproot.open(file_path) as f_in:
        tree = f_in["events"]
        n_entries = tree.num_entries
        if max_events:
            n_entries = min(n_entries, max_events)
        
        chunk_size = 500
        for start in range(0, n_entries, chunk_size):
            end = min(start + chunk_size, n_entries)
            arrays = tree.arrays(branches, entry_start=start, entry_stop=end)
            
            arrays["idx_reco"] = arrays["_RecoMCTruthLink_from.index"]
            arrays["idx_mc"] = arrays["_RecoMCTruthLink_to.index"]
            arrays["mc_weight"] = arrays["RecoMCTruthLink.weight"]
            
            reco_particles, reco_particles_p4 = pfl.RecoParticleFilter(arrays=arrays, p_type="PandoraPFOs").results
            mc_particles, mc_particles_p4 = pfl.MCParticleFilter(arrays=arrays, p_type="MCParticles").results
            
            # Cluster jets
            reco_jets, reco_constituent_indices = cl.RecoJetClusterer(particles=reco_particles, particles_p4=reco_particles_p4).results
            
            if len(reco_jets) == 0:
                continue
            
            # For gen-level info, we extract the visible Higgs mass by tracing stable descendants of tau leptons
            if is_signal:
                # 1. Find all tau leptons (PDG 15)
                pdgs = arrays["MCParticles.PDG"]
                is_tau = (abs(pdgs) == 15)
                
                # We need to trace descendants. To simplify, let's just find all stable particles (status 1)
                # that are not neutrinos and sum them to get the total visible 4-momentum of the event.
                # In ZH events where Z -> invisible or Z -> jets, we might need more care.
                # However, the user asked for gen-level visible Higgs mass.
                # Let's use the stable daughters of the taus.
                
                mc_p4 = mc_particles_p4 # Status 1, no neutrinos
                
                # Simple approximation: In H -> tautau events, the Higgs visible mass is the invariant mass of all
                # stable non-neutrino particles that don't come from the Z.
                # A better way is to find the two taus and sum their visible descendants.
                
                # Let's implement a simplified but robust version:
                # Find all MCParticles that are descendants of a tau.
                
                event_gen_higgs_masses = []
                for i_ev in range(len(arrays)):
                    ev_pdgs = arrays["MCParticles.PDG"][i_ev]
                    ev_parents_begin = arrays["MCParticles.parents_begin"][i_ev]
                    ev_parents_end = arrays["MCParticles.parents_end"][i_ev]
                    ev_parents_idx = arrays["_MCParticles_parents.index"][i_ev]
                    
                    # Find indices of tau leptons
                    tau_indices = np.where(abs(ev_pdgs) == 15)[0]
                    
                    # For each tau, find all its descendants that are stable and visible
                    tau_vis_p4s = []
                    for t_idx in tau_indices:
                        descendants = []
                        to_process = [t_idx]
                        while to_process:
                            curr = to_process.pop()
                            d_begin = arrays["MCParticles.daughters_begin"][i_ev][curr]
                            d_end = arrays["MCParticles.daughters_end"][i_ev][curr]
                            d_indices = arrays["_MCParticles_daughters.index"][i_ev][d_begin:d_end]
                            for d_idx in d_indices:
                                if arrays["MCParticles.generatorStatus"][i_ev][d_idx] == 1:
                                    # Stable. Is it visible? (Not a neutrino)
                                    d_pdg = abs(arrays["MCParticles.PDG"][i_ev][d_idx])
                                    if d_pdg not in [12, 14, 16]:
                                        descendants.append(d_idx)
                                else:
                                    to_process.append(d_idx)
                        
                        if descendants:
                            # Sum p4 of visible descendants
                            ev_px = arrays["MCParticles.momentum.x"][i_ev][descendants]
                            ev_py = arrays["MCParticles.momentum.y"][i_ev][descendants]
                            ev_pz = arrays["MCParticles.momentum.z"][i_ev][descendants]
                            ev_m = arrays["MCParticles.mass"][i_ev][descendants]
                            
                            # E = sqrt(p^2 + m^2)
                            ev_e = np.sqrt(ev_px**2 + ev_py**2 + ev_pz**2 + ev_m**2)
                            
                            total_p4 = vector.obj(
                                px=ak.sum(ev_px),
                                py=ak.sum(ev_py),
                                pz=ak.sum(ev_pz),
                                energy=ak.sum(ev_e)
                            )
                            tau_vis_p4s.append(total_p4)
                    
                    if len(tau_vis_p4s) == 2:
                        m_gen = (tau_vis_p4s[0] + tau_vis_p4s[1]).mass
                        event_gen_higgs_masses.append(m_gen)
                    else:
                        event_gen_higgs_masses.append(-1)
            
            # Collect all jet pts before any selection
            file_jet_pts.extend(ak.to_numpy(ak.flatten(reco_jets.pt)))

            valid_particle_mask = reco_particles["PDG"] != 0
            all_particle_lifetime_info = lt.find_all_track_pcas(
                events=arrays,
                reco_particle_collection="PandoraPFOs",
                track_collection="SiTracks_Refitted",
                vertex_collection="PrimaryVertices",
                valid_particle_mask=valid_particle_mask
            )
            
            lifetime_vars = ["dxy", "dz", "dxy_error", "dz_error"]
            lifetime_info = lt.assign_lifetime_vars_to_jets(
                all_particle_lifetime_info=all_particle_lifetime_info,
                reco_jet_constituent_indices=reco_constituent_indices,
                reco_jets=reco_jets,
                lifetime_vars=lifetime_vars
            )
            
            num_ptcls_per_jet = ak.num(reco_constituent_indices, axis=-1)
            reco_cand_p4s = g.get_jet_constituent_property(reco_particles_p4, reco_constituent_indices, num_ptcls_per_jet)
            reco_cand_p4s = g.reinitialize_p4(reco_cand_p4s)
            
            def get_candid_from_pdg(pdg_ids, charges):
                flat_charges = ak.flatten(np.abs(charges) > 0)
                flat_np_pdg = ak.to_numpy(ak.flatten(pdg_ids))
                is_hadron = np.vectorize(lambda p: abs(p) > 20 and abs(p) != 22)(flat_np_pdg)
                candid_np = np.where(is_hadron, np.where(np.abs(flat_charges) > 0, 211, 130), np.abs(flat_np_pdg))
                return ak.unflatten(candid_np, ak.num(pdg_ids))

            reco_cand_pdgs_all = get_candid_from_pdg(reco_particles.PDG, reco_particles.charge)
            reco_cand_pdgs = g.get_jet_constituent_property(reco_cand_pdgs_all, reco_constituent_indices, num_ptcls_per_jet)
            reco_cand_charges = g.get_jet_constituent_property(reco_particles.charge, reco_constituent_indices, num_ptcls_per_jet)
            
            for i_ev in range(len(reco_jets)):
                ev_jets = reco_jets[i_ev]
                if len(ev_jets) == 0:
                    continue
                
                ev_cand_p4s = reco_cand_p4s[i_ev]
                ev_cand_pdgs = reco_cand_pdgs[i_ev]
                ev_cand_charges = reco_cand_charges[i_ev]
                ev_lifetime = {k: lifetime_info[k][i_ev] for k in lifetime_vars}
                
                n_jets = len(ev_jets)
                eps = 1e-6
                all_jet_features = []
                all_jet_kinematics = []
                all_jet_masks = []
                
                for i_jet in range(n_jets):
                    jet_p4 = ev_jets[i_jet]
                    cands_p4 = ev_cand_p4s[i_jet]
                    cands_pdg = ak.to_numpy(ev_cand_pdgs[i_jet])
                    cands_charge = ak.to_numpy(ev_cand_charges[i_jet])
                    
                    cand_deta = ak.to_numpy(f.signedDeltaEta(cands_p4.eta, jet_p4.eta))
                    cand_dphi = ak.to_numpy(f.signedDeltaPhi(cands_p4.phi, jet_p4.phi))
                    cand_logpt = np.log(np.maximum(ak.to_numpy(cands_p4.pt), eps))
                    cand_loge = np.log(np.maximum(ak.to_numpy(cands_p4.energy), eps))
                    cand_logptrel = np.log(np.maximum(ak.to_numpy(cands_p4.pt) / jet_p4.pt, eps))
                    cand_logerel = np.log(np.maximum(ak.to_numpy(cands_p4.energy) / jet_p4.energy, eps))
                    cand_deltaR = ak.to_numpy(f.deltaR_etaPhi(cands_p4.eta, cands_p4.phi, jet_p4.eta, jet_p4.phi))
                    
                    isElectron = (cands_pdg == 11).astype(np.float32)
                    isMuon = (cands_pdg == 13).astype(np.float32)
                    isPhoton = (cands_pdg == 22).astype(np.float32)
                    isChargedHadron = (cands_pdg == 211).astype(np.float32)
                    isNeutralHadron = (cands_pdg == 130).astype(np.float32)
                    
                    dz = ak.to_numpy(ev_lifetime["dz"][i_jet])
                    dz_err = ak.to_numpy(ev_lifetime["dz_error"][i_jet])
                    dxy = ak.to_numpy(ev_lifetime["dxy"][i_jet])
                    dxy_err = ak.to_numpy(ev_lifetime["dxy_error"][i_jet])
                    dz_sig = dz / np.maximum(dz_err, eps)
                    dxy_sig = dxy / np.maximum(dxy_err, eps)
                    
                    feats = np.stack([
                        cand_deta, cand_dphi, cand_logpt, cand_loge, cand_logptrel, cand_logerel, cand_deltaR, cands_charge,
                        isElectron, isMuon, isPhoton, isChargedHadron, isNeutralHadron,
                        dz, dz_sig, dxy, dxy_sig
                    ], axis=-1)
                    
                    kinematics_pxpypze = np.stack([
                        ak.to_numpy(cands_p4.px), ak.to_numpy(cands_p4.py), ak.to_numpy(cands_p4.pz), ak.to_numpy(cands_p4.energy)
                    ], axis=-1)
                    
                    n_cands = len(feats)
                    if n_cands > MAX_CANDS:
                        feats = feats[:MAX_CANDS]
                        kinematics_pxpypze = kinematics_pxpypze[:MAX_CANDS]
                        mask = np.ones(MAX_CANDS, dtype=bool)
                    else:
                        mask = np.zeros(MAX_CANDS, dtype=bool)
                        mask[:n_cands] = True
                        padding_feats = np.zeros((MAX_CANDS - n_cands, 17), dtype=np.float32)
                        feats = np.concatenate([feats, padding_feats], axis=0)
                        padding_kin = np.zeros((MAX_CANDS - n_cands, 4), dtype=np.float32)
                        kinematics_pxpypze = np.concatenate([kinematics_pxpypze, padding_kin], axis=0)
                    all_jet_features.append(feats)
                    all_jet_kinematics.append(kinematics_pxpypze)
                    all_jet_masks.append(mask)
                
                feat_tensor = torch.from_numpy(np.stack(all_jet_features)).float().to(device)
                kin_tensor = torch.from_numpy(np.stack(all_jet_kinematics)).float().to(device)
                mask_tensor = torch.from_numpy(np.stack(all_jet_masks)).bool().to(device).unsqueeze(1)
                
                feat_tensor = feat_tensor.transpose(1, 2)
                kin_tensor = kin_tensor.transpose(1, 2)
                
                with torch.no_grad():
                    preds = model.ParTau(feat_tensor, kin_tensor, mask_tensor)
                
                logits = preds["is_tau"].cpu().numpy()
                scores = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                tau_scores = scores[:, 1]
                charges = torch.sigmoid(preds["charge"]).cpu().numpy().flatten()
                kinematics = preds["kinematics"].cpu().numpy()
                
                tau_indices = np.where(tau_scores > TAU_ID_THRESHOLD)[0]
                if len(tau_indices) == 2:
                    idx1, idx2 = tau_indices
                    q1 = 1 if charges[idx1] > 0.5 else -1
                    q2 = 1 if charges[idx2] > 0.5 else -1
                    if q1 * q2 < 0:
                        p4_1 = decode_kinematic_predictions(kinematics[idx1:idx1+1], ev_jets[idx1:idx1+1])[0]
                        p4_2 = decode_kinematic_predictions(kinematics[idx2:idx2+1], ev_jets[idx2:idx2+1])[0]
                        file_masses.append((p4_1 + p4_2).mass)
                        
                        if is_signal:
                            file_gen_masses.append(event_gen_higgs_masses[i_ev])
    return file_masses, file_jet_pts, file_gen_masses

def process_files_parallel(file_paths, ckpt_path, cfg, device_str, max_events, n_workers, is_signal=False):
    args = [(fp, ckpt_path, cfg, device_str, max_events, is_signal) for fp in file_paths]
    all_masses = []
    all_gen_masses = []
    all_jet_pts = []
    
    # Use spawn for CUDA compatibility if using GPU
    ctx = mp.get_context('spawn') if 'cuda' in device_str else mp.get_context('fork')
    
    with ctx.Pool(processes=n_workers) as pool:
        for masses, jet_pts, gen_masses in tqdm(pool.imap_unordered(process_single_file, args), total=len(file_paths), desc="Processing files"):
            all_masses.extend(masses)
            all_jet_pts.extend(jet_pts)
            all_gen_masses.extend(gen_masses)
            
    return np.array(all_masses), np.array(all_jet_pts), np.array(all_gen_masses)

def main():
    parser = argparse.ArgumentParser(description="Reconstruct Higgs to tautau peak.")
    parser.add_argument("--ckpt", default="outputs/0612_multipartau_full_b8483f6/models/ParT-model_best.ckpt")
    parser.add_argument("--n-files", type=int, default=5, help="Number of files per sample to process")
    parser.add_argument("--max-events", type=int, default=None, help="Max events per file")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str} with {args.n_workers} workers")
    
    # Load config once to pass to workers
    with initialize(version_base=None, config_path="../../mltau/config"):
        cfg = compose(config_name="main")
    
    # Samples
    signal_dir = "/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ZH_Htautau_ecm365/root"
    ttbar_dir = "/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root"
    qq_dir = "/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm365/root"
    
    signal_files = sorted(glob.glob(os.path.join(signal_dir, "**/*.root"), recursive=True))[:args.n_files]
    ttbar_files = sorted(glob.glob(os.path.join(ttbar_dir, "**/*.root"), recursive=True))[:args.n_files]
    qq_files = sorted(glob.glob(os.path.join(qq_dir, "**/*.root"), recursive=True))[:args.n_files]
    
    print(f"Found {len(signal_files)} signal files, {len(ttbar_files)} ttbar files, and {len(qq_files)} qq files.")
    
    print("Processing signal...")
    signal_masses, signal_jet_pts, signal_gen_masses = process_files_parallel(signal_files, args.ckpt, cfg, device_str, args.max_events, args.n_workers, is_signal=True)
    
    print("Processing ttbar background...")
    ttbar_masses, ttbar_jet_pts, _ = process_files_parallel(ttbar_files, args.ckpt, cfg, device_str, args.max_events, args.n_workers, is_signal=False)

    print("Processing qq background...")
    qq_masses, qq_jet_pts, _ = process_files_parallel(qq_files, args.ckpt, cfg, device_str, args.max_events, args.n_workers, is_signal=False)
    
    # Plotting
    hep.style.use("CMS")
    
    # 1. Higgs Peak Plot
    plt.figure(figsize=(10, 8))
    bins_m = np.linspace(0, 200, 50)
    plt.hist(signal_masses, bins=bins_m, histtype="step", label=r"$ZH \to Z\tau\tau$ (Signal)", color="blue", lw=2)
    plt.hist(ttbar_masses, bins=bins_m, histtype="step", label=r"$t\bar{t}$ (Background)", color="red", lw=2)
    plt.hist(qq_masses, bins=bins_m, histtype="step", label=r"$q\bar{q}$ (Background)", color="green", lw=2)
    plt.xlabel(r"Visible $M_{\tau\tau}$ [GeV]")
    plt.ylabel("Events")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Higgs to tautau peak reconstruction (ParTau)")
    plt.savefig("higgs_peak.png")
    print(f"Higgs peak plot saved to higgs_peak.png")

    # 2. All Jets Pt Plot
    plt.figure(figsize=(10, 8))
    bins_pt = np.linspace(0, 150, 60)
    plt.hist(signal_jet_pts, bins=bins_pt, histtype="step", label=r"$ZH \to Z\tau\tau$ (All Jets)", color="blue", lw=2, density=True)
    plt.hist(ttbar_jet_pts, bins=bins_pt, histtype="step", label=r"$t\bar{t}$ (All Jets)", color="red", lw=2, density=True)
    plt.hist(qq_jet_pts, bins=bins_pt, histtype="step", label=r"$q\bar{q}$ (All Jets)", color="green", lw=2, density=True)
    plt.xlabel(r"Jet $p_T$ [GeV]")
    plt.ylabel("Normalized Yield")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("All clustered jets before selection")
    plt.savefig("all_jets_pt.png")
    print(f"All jets pt plot saved to all_jets_pt.png")
    
    # 3. 2D Plot for Signal: Gen vs Reco
    valid_mask = (signal_gen_masses > 0)
    if np.sum(valid_mask) > 0:
        plt.figure(figsize=(10, 8))
        plt.hist2d(signal_gen_masses[valid_mask], signal_masses[valid_mask], bins=(50, 50), range=[[0, 150], [0, 150]], cmap="Blues")
        plt.colorbar(label="Events")
        plt.plot([0, 150], [0, 150], color="red", linestyle="--")
        plt.xlabel(r"Generated Visible $M_{\tau\tau}$ [GeV]")
        plt.ylabel(r"Reconstructed Visible $M_{\tau\tau}$ [GeV]")
        plt.title(r"$ZH \to Z\tau\tau$: Gen vs Reco Visible Mass")
        plt.savefig("gen_vs_reco_mass.png")
        print(f"2D Gen vs Reco plot saved to gen_vs_reco_mass.png")
    
    print(f"Signal events selected: {len(signal_masses)}")
    print(f"ttbar events selected: {len(ttbar_masses)}")
    print(f"qq events selected: {len(qq_masses)}")
    print(f"Total jets processed: {len(signal_jet_pts) + len(ttbar_jet_pts) + len(qq_jet_pts)}")

if __name__ == "__main__":
    main()
