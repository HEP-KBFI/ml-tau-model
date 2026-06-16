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
    
    # Selection stages
    file_masses_id05 = []
    file_masses_id09 = []
    file_masses_id09os = []
    
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
            
            reco_jets, reco_constituent_indices = cl.RecoJetClusterer(particles=reco_particles, particles_p4=reco_particles_p4).results
            if len(reco_jets) == 0: continue
            
            if is_signal:
                event_gen_higgs_masses = []
                for i_ev in range(len(arrays)):
                    ev_pdgs = arrays["MCParticles.PDG"][i_ev]
                    tau_indices = np.where(abs(ev_pdgs) == 15)[0]
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
                                    d_pdg = abs(arrays["MCParticles.PDG"][i_ev][d_idx])
                                    if d_pdg not in [12, 14, 16]: descendants.append(d_idx)
                                else: to_process.append(d_idx)
                        if descendants:
                            ev_px = arrays["MCParticles.momentum.x"][i_ev][descendants]; ev_py = arrays["MCParticles.momentum.y"][i_ev][descendants]
                            ev_pz = arrays["MCParticles.momentum.z"][i_ev][descendants]; ev_m = arrays["MCParticles.mass"][i_ev][descendants]
                            ev_e = np.sqrt(ev_px**2 + ev_py**2 + ev_pz**2 + ev_m**2)
                            total_p4 = vector.obj(px=ak.sum(ev_px), py=ak.sum(ev_py), pz=ak.sum(ev_pz), energy=ak.sum(ev_e))
                            tau_vis_p4s.append(total_p4)
                    if len(tau_vis_p4s) == 2: event_gen_higgs_masses.append((tau_vis_p4s[0] + tau_vis_p4s[1]).mass)
                    else: event_gen_higgs_masses.append(-1)
            
            file_jet_pts.extend(ak.to_numpy(ak.flatten(reco_jets.pt)))
            valid_particle_mask = reco_particles["PDG"] != 0
            all_particle_lifetime_info = lt.find_all_track_pcas(events=arrays, reco_particle_collection="PandoraPFOs", track_collection="SiTracks_Refitted", vertex_collection="PrimaryVertices", valid_particle_mask=valid_particle_mask)
            lifetime_vars = ["dxy", "dz", "dxy_error", "dz_error"]
            lifetime_info = lt.assign_lifetime_vars_to_jets(all_particle_lifetime_info=all_particle_lifetime_info, reco_jet_constituent_indices=reco_constituent_indices, reco_jets=reco_jets, lifetime_vars=lifetime_vars)
            num_ptcls_per_jet = ak.num(reco_constituent_indices, axis=-1)
            reco_cand_p4s = g.reinitialize_p4(g.get_jet_constituent_property(reco_particles_p4, reco_constituent_indices, num_ptcls_per_jet))
            
            def get_candid_from_pdg(pdg_ids, charges):
                flat_charges = ak.flatten(np.abs(charges) > 0); flat_np_pdg = ak.to_numpy(ak.flatten(pdg_ids))
                is_hadron = np.vectorize(lambda p: abs(p) > 20 and abs(p) != 22)(flat_np_pdg)
                candid_np = np.where(is_hadron, np.where(np.abs(flat_charges) > 0, 211, 130), np.abs(flat_np_pdg))
                return ak.unflatten(candid_np, ak.num(pdg_ids))

            reco_cand_pdgs = g.get_jet_constituent_property(get_candid_from_pdg(reco_particles.PDG, reco_particles.charge), reco_constituent_indices, num_ptcls_per_jet)
            reco_cand_charges = g.get_jet_constituent_property(reco_particles.charge, reco_constituent_indices, num_ptcls_per_jet)
            
            for i_ev in range(len(reco_jets)):
                ev_jets = reco_jets[i_ev]
                if len(ev_jets) == 0: continue
                ev_cand_p4s = reco_cand_p4s[i_ev]; ev_cand_pdgs = reco_cand_pdgs[i_ev]; ev_cand_charges = reco_cand_charges[i_ev]
                ev_lifetime = {k: lifetime_info[k][i_ev] for k in lifetime_vars}
                n_jets = len(ev_jets); eps = 1e-6
                all_jet_features = []; all_jet_kinematics = []; all_jet_masks = []
                for i_jet in range(n_jets):
                    jet_p4 = ev_jets[i_jet]; cands_p4 = ev_cand_p4s[i_jet]; cands_pdg = ak.to_numpy(ev_cand_pdgs[i_jet]); cands_charge = ak.to_numpy(ev_cand_charges[i_jet])
                    cand_deta = ak.to_numpy(f.signedDeltaEta(cands_p4.eta, jet_p4.eta)); cand_dphi = ak.to_numpy(f.signedDeltaPhi(cands_p4.phi, jet_p4.phi))
                    cand_logpt = np.log(np.maximum(ak.to_numpy(cands_p4.pt), eps)); cand_loge = np.log(np.maximum(ak.to_numpy(cands_p4.energy), eps))
                    cand_logptrel = np.log(np.maximum(ak.to_numpy(cands_p4.pt) / jet_p4.pt, eps)); cand_logerel = np.log(np.maximum(ak.to_numpy(cands_p4.energy) / jet_p4.energy, eps))
                    cand_deltaR = ak.to_numpy(f.deltaR_etaPhi(cands_p4.eta, cands_p4.phi, jet_p4.eta, jet_p4.phi))
                    isElectron = (cands_pdg == 11).astype(np.float32); isMuon = (cands_pdg == 13).astype(np.float32); isPhoton = (cands_pdg == 22).astype(np.float32); isChargedHadron = (cands_pdg == 211).astype(np.float32); isNeutralHadron = (cands_pdg == 130).astype(np.float32)
                    dz = ak.to_numpy(ev_lifetime["dz"][i_jet]); dz_err = ak.to_numpy(ev_lifetime["dz_error"][i_jet]); dxy = ak.to_numpy(ev_lifetime["dxy"][i_jet]); dxy_err = ak.to_numpy(ev_lifetime["dxy_error"][i_jet])
                    feats = np.stack([cand_deta, cand_dphi, cand_logpt, cand_loge, cand_logptrel, cand_logerel, cand_deltaR, cands_charge, isElectron, isMuon, isPhoton, isChargedHadron, isNeutralHadron, dz, dz / np.maximum(dz_err, eps), dxy, dxy / np.maximum(dxy_err, eps)], axis=-1)
                    kin_pxpypze = np.stack([ak.to_numpy(cands_p4.px), ak.to_numpy(cands_p4.py), ak.to_numpy(cands_p4.pz), ak.to_numpy(cands_p4.energy)], axis=-1)
                    n_cands = len(feats)
                    if n_cands > MAX_CANDS:
                        feats = feats[:MAX_CANDS]; kin_pxpypze = kin_pxpypze[:MAX_CANDS]; mask = np.ones(MAX_CANDS, dtype=bool)
                    else:
                        mask = np.zeros(MAX_CANDS, dtype=bool); mask[:n_cands] = True
                        feats = np.concatenate([feats, np.zeros((MAX_CANDS - n_cands, 17), dtype=np.float32)], axis=0); kin_pxpypze = np.concatenate([kin_pxpypze, np.zeros((MAX_CANDS - n_cands, 4), dtype=np.float32)], axis=0)
                    all_jet_features.append(feats); all_jet_kinematics.append(kin_pxpypze); all_jet_masks.append(mask)
                
                feat_tensor = torch.from_numpy(np.stack(all_jet_features)).float().to(device).transpose(1, 2)
                kin_tensor = torch.from_numpy(np.stack(all_jet_kinematics)).float().to(device).transpose(1, 2)
                mask_tensor = torch.from_numpy(np.stack(all_jet_masks)).bool().to(device).unsqueeze(1)
                with torch.no_grad(): preds = model.ParTau(feat_tensor, kin_tensor, mask_tensor)
                logits = preds["is_tau"].cpu().numpy(); scores = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True); tau_scores = scores[:, 1]
                charges = torch.sigmoid(preds["charge"]).cpu().numpy().flatten(); kinematics = preds["kinematics"].cpu().numpy()
                
                idx05 = np.where(tau_scores > 0.5)[0]; idx09 = np.where(tau_scores > 0.9)[0]
                if len(idx05) == 2:
                    p4_1 = decode_kinematic_predictions(kinematics[idx05[0]:idx05[0]+1], ev_jets[idx05[0]:idx05[0]+1])[0]
                    p4_2 = decode_kinematic_predictions(kinematics[idx05[1]:idx05[1]+1], ev_jets[idx05[1]:idx05[1]+1])[0]
                    file_masses_id05.append((p4_1 + p4_2).mass)
                if len(idx09) == 2:
                    p4_1 = decode_kinematic_predictions(kinematics[idx09[0]:idx09[0]+1], ev_jets[idx09[0]:idx09[0]+1])[0]
                    p4_2 = decode_kinematic_predictions(kinematics[idx09[1]:idx09[1]+1], ev_jets[idx09[1]:idx09[1]+1])[0]
                    m_reco = (p4_1 + p4_2).mass; file_masses_id09.append(m_reco)
                    if (1 if charges[idx09[0]] > 0.5 else -1) * (1 if charges[idx09[1]] > 0.5 else -1) < 0:
                        file_masses_id09os.append(m_reco)
                        if is_signal: file_gen_masses.append(event_gen_higgs_masses[i_ev])
    return (file_masses_id05, file_masses_id09, file_masses_id09os), file_jet_pts, file_gen_masses

def process_files_parallel(file_paths, ckpt_path, cfg, device_str, max_events, n_workers, is_signal=False):
    args = [(fp, ckpt_path, cfg, device_str, max_events, is_signal) for fp in file_paths]
    all_masses = {"05": [], "09": [], "09os": []}
    all_gen_masses = []; all_jet_pts = []
    ctx = mp.get_context('spawn') if 'cuda' in device_str else mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        for (m05, m09, m09os), jet_pts, gen_masses in tqdm(pool.imap_unordered(process_single_file, args), total=len(file_paths), desc="Processing files"):
            all_masses["05"].extend(m05); all_masses["09"].extend(m09); all_masses["09os"].extend(m09os)
            all_jet_pts.extend(jet_pts); all_gen_masses.extend(gen_masses)
    return all_masses, np.array(all_jet_pts), np.array(all_gen_masses)

def main():
    parser = argparse.ArgumentParser(description="Reconstruct Higgs to tautau peak.")
    parser.add_argument("--ckpt", default="outputs/0612_multipartau_full_b8483f6/models/ParT-model_best.ckpt")
    parser.add_argument("--n-files", type=int, default=5); parser.add_argument("--max-events", type=int, default=None); parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args(); device_str = "cuda" if torch.cuda.is_available() else "cpu"
    with initialize(version_base=None, config_path="../../mltau/config"): cfg = compose(config_name="main")
    
    s_dir = "/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ZH_Htautau_ecm365/root"
    t_dir = "/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root"
    q_dir = "/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm365/root"
    
    s_files = sorted(glob.glob(os.path.join(s_dir, "**/*.root"), recursive=True))[:args.n_files]
    t_files = sorted(glob.glob(os.path.join(t_dir, "**/*.root"), recursive=True))[:args.n_files]
    q_files = sorted(glob.glob(os.path.join(q_dir, "**/*.root"), recursive=True))[:args.n_files]
    
    print("Processing signal..."); sig_m, sig_pt, sig_gen = process_files_parallel(s_files, args.ckpt, cfg, device_str, args.max_events, args.n_workers, is_signal=True)
    print("Processing ttbar background..."); tt_m, tt_pt, _ = process_files_parallel(t_files, args.ckpt, cfg, device_str, args.max_events, args.n_workers, is_signal=False)
    print("Processing qq background..."); qq_m, qq_pt, _ = process_files_parallel(q_files, args.ckpt, cfg, device_str, args.max_events, args.n_workers, is_signal=False)
    
    hep.style.use("CMS")
    plt.figure(figsize=(12, 10)); bins_m = np.linspace(0, 200, 50)
    plt.hist(sig_m["05"], bins=bins_m, histtype="step", label="Sig (ID > 0.5)", color="blue", ls=":")
    plt.hist(sig_m["09"], bins=bins_m, histtype="step", label="Sig (ID > 0.9)", color="blue", ls="--")
    plt.hist(sig_m["09os"], bins=bins_m, histtype="step", label="Sig (ID > 0.9+OS)", color="blue", lw=2)
    plt.hist(tt_m["05"], bins=bins_m, histtype="step", label="ttbar (ID > 0.5)", color="red", ls=":")
    plt.hist(tt_m["09"], bins=bins_m, histtype="step", label="ttbar (ID > 0.9)", color="red", ls="--")
    plt.hist(tt_m["09os"], bins=bins_m, histtype="step", label="ttbar (ID > 0.9+OS)", color="red", lw=2)
    plt.hist(qq_m["05"], bins=bins_m, histtype="step", label="qq (ID > 0.5)", color="green", ls=":")
    plt.hist(qq_m["09"], bins=bins_m, histtype="step", label="qq (ID > 0.9)", color="green", ls="--")
    plt.hist(qq_m["09os"], bins=bins_m, histtype="step", label="qq (ID > 0.9+OS)", color="green", lw=2)
    plt.xlabel(r"Visible $M_{\tau\tau}$ [GeV]"); plt.ylabel("Events"); plt.legend(ncol=3); plt.grid(alpha=0.3); plt.title("Selection Cut Flow"); plt.savefig("higgs_peak_cutflow.png")
    
    plt.figure(figsize=(10, 8)); bins_pt = np.linspace(0, 150, 60)
    plt.hist(sig_pt, bins=bins_pt, histtype="step", label="ZH", color="blue", lw=2, density=True)
    plt.hist(tt_pt, bins=bins_pt, histtype="step", label="ttbar", color="red", lw=2, density=True)
    plt.hist(qq_pt, bins=bins_pt, histtype="step", label="qq", color="green", lw=2, density=True)
    plt.xlabel(r"Jet $p_T$ [GeV]"); plt.ylabel("Normalized Yield"); plt.legend(); plt.grid(alpha=0.3); plt.savefig("all_jets_pt.png")
    
    if np.sum(sig_gen > 0) > 0:
        plt.figure(figsize=(10, 8)); plt.hist2d(sig_gen[sig_gen > 0], np.array(sig_m["09os"])[sig_gen > 0], bins=(50, 50), range=[[0, 150], [0, 150]], cmap="Blues")
        plt.colorbar(label="Events"); plt.plot([0, 150], [0, 150], color="red", ls="--"); plt.xlabel("Gen Visible Mass [GeV]"); plt.ylabel("Reco Visible Mass [GeV]"); plt.savefig("gen_vs_reco_mass.png")

if __name__ == "__main__":
    main()
