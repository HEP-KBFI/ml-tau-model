
import awkward as ak
import numpy as np
import vector
import pandas as pd
from particle import pdgid

# Add project paths to import existing tools
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "ml-tau-data"))
from ntupelizer.tools import tau_decaymode as dm

vector.register_awkward()

def map_pdgid_to_candid(pdg_id):
    if pdgid.is_hadron(pdg_id):
        if abs(pdgid.charge(pdg_id)) > 0:
            return 211  # charged hadron
        else:
            return 130  # neutral hadron
    elif abs(pdg_id) == 11: return 11
    elif abs(pdg_id) == 13: return 13
    elif abs(pdg_id) == 22: return 22
    return abs(pdg_id)

def find_tau_daughters_pure_indices(event, tau_idx):
    pids = event["Gen_Part_PID"]
    st = event["Gen_Part_Status"]
    d1 = event["Gen_Part_D1"]
    d2 = event["Gen_Part_D2"]
    
    daughters = []
    stack = [tau_idx]
    visited = set()
    
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        
        start = d1[idx]
        end = d2[idx]
        
        if start == -1:
            # Leaf node
            daughters.append(idx)
        else:
            for d in range(start, end + 1):
                if d < len(pids):
                    stack.append(d)
    return daughters

def get_visible_tau_info(event, tau_idx):
    pids = event["Gen_Part_PID"]
    st = event["Gen_Part_Status"]
    pt = event["Gen_Part_PT"]
    eta = event["Gen_Part_Eta"]
    phi = event["Gen_Part_Phi"]
    mass = event["Gen_Part_Mass"]
    
    daughter_indices = find_tau_daughters_pure_indices(event, tau_idx)
    
    vis_p4 = vector.obj(pt=0, eta=0, phi=0, mass=0)
    daughter_pids = []
    
    for i in daughter_indices:
        p = vector.obj(pt=pt[i], eta=eta[i], phi=phi[i], mass=mass[i]/1000.0)
        pid = abs(pids[i])
        
        # Add to visible p4 if it's stable and not a neutrino
        if st[i] == 1 and pid not in [12, 14, 16]:
            vis_p4 = vis_p4 + p
            
        # Add to daughter PIDs for decay mode calculation
        if st[i] == 1 and pid not in [12, 14, 16, 22]:
            daughter_pids.append(pids[i])
        elif pid == 111:
            daughter_pids.append(pids[i])
            
    if vis_p4.pt == 0:
        return None
        
    candid_pids = [map_pdgid_to_candid(p) for p in daughter_pids]
    raw_dm = dm.get_decaymode(candid_pids)
    
    return {
        "p4": vis_p4,
        "charge": np.sign(pids[tau_idx]),
        "decay_mode": raw_dm
    }

def process_file(input_path, output_path, max_events=None):
    print(f"Processing {input_path} ...")
    df = ak.from_parquet(input_path)
    
    num_events = len(df) if max_events is None else min(max_events, len(df))
    all_jets = []
    
    for i in range(num_events):
        if i % 100 == 0:
            print(f"Event {i}/{num_events}")
            
        event = df[i]
        
        # 1. Find signal taus
        pids = event["Gen_Part_PID"]
        tau_indices = np.where((np.abs(pids) == 15) & (event["Gen_Part_Status"] == 23))[0]
        
        signal_taus = []
        for t_idx in tau_indices:
            info = get_visible_tau_info(event, t_idx)
            if info:
                signal_taus.append(info)
            
        # 2. Process jets
        jet_pts = event["FullReco_JetAK4_PT"]
        jet_etas = event["FullReco_JetAK4_Eta"]
        jet_phis = event["FullReco_JetAK4_Phi"]
        jet_masses = event["FullReco_JetAK4_Mass"]
        jet_const_idx = event["FullReco_JetAK4_ConstituentsIdx"]
        
        # PF particles
        pf_pt = event["FullReco_PFPart_PT"]
        pf_eta = event["FullReco_PFPart_Eta"]
        pf_phi = event["FullReco_PFPart_Phi"]
        pf_e = event["FullReco_PFPart_E"]
        pf_charge = event["FullReco_PFPart_Charge"]
        pf_pid = event["FullReco_PFPart_PID"]
        pf_dz = event["FullReco_PFPart_DZ"]
        pf_dz_err = event["FullReco_PFPart_ErrorDZ"]
        pf_d0 = event["FullReco_PFPart_D0"]
        pf_d0_err = event["FullReco_PFPart_ErrorD0"]
        
        for j in range(len(jet_pts)):
            jet_p4 = vector.obj(pt=jet_pts[j], eta=jet_etas[j], phi=jet_phis[j], mass=jet_masses[j])
            
            # Match to signal tau
            matched_tau = None
            min_dr = 0.4
            for tau in signal_taus:
                dr = jet_p4.deltaR(tau["p4"])
                if dr < min_dr:
                    min_dr = dr
                    matched_tau = tau
            
            is_tau = 1 if matched_tau else 0
            
            # Get constituents
            c_indices = jet_const_idx[j]
            constituents = []
            for ci in c_indices:
                pid = pf_pid[ci]
                is_ele = 1 if abs(pid) == 11 else 0
                is_mu = 1 if abs(pid) == 13 else 0
                is_pho = 1 if abs(pid) == 22 else 0
                is_had = pdgid.is_hadron(pid)
                is_ch = 1 if (is_had and abs(pf_charge[ci]) > 0) else 0
                is_nh = 1 if (is_had and abs(pf_charge[ci]) == 0) else 0
                
                deta = pf_eta[ci] - jet_etas[j]
                dphi = pf_phi[ci] - jet_phis[j]
                while dphi > np.pi: dphi -= 2*np.pi
                while dphi < -np.pi: dphi += 2*np.pi
                
                constituents.append({
                    "cand_deta": deta,
                    "cand_dphi": dphi,
                    "cand_logpt": np.log(max(pf_pt[ci], 1e-3)),
                    "cand_loge": np.log(max(pf_e[ci], 1e-3)),
                    "cand_logptrel": np.log(max(pf_pt[ci] / jet_pts[j], 1e-6)),
                    "cand_logerel": np.log(max(pf_e[ci] / (jet_p4.energy), 1e-6)),
                    "cand_deltaR": np.sqrt(deta**2 + dphi**2),
                    "cand_charge": pf_charge[ci],
                    "isElectron": is_ele,
                    "isMuon": is_mu,
                    "isPhoton": is_pho,
                    "isChargedHadron": is_ch,
                    "isNeutralHadron": is_nh,
                    "cand_dz": pf_dz[ci],
                    "cand_dz_error": pf_dz_err[ci],
                    "cand_dxy": pf_d0[ci],
                    "cand_dxy_error": pf_d0_err[ci]
                })
            
            max_cands = 20
            constituents = sorted(constituents, key=lambda x: x["cand_logpt"], reverse=True)[:max_cands]
            while len(constituents) < max_cands:
                constituents.append({f: 0 for f in [
                    "cand_deta", "cand_dphi", "cand_logpt", "cand_loge", "cand_logptrel", "cand_logerel",
                    "cand_deltaR", "cand_charge", "isElectron", "isMuon", "isPhoton", "isChargedHadron",
                    "isNeutralHadron", "cand_dz", "cand_dz_error", "cand_dxy", "cand_dxy_error"
                ]})

            jet_record = {
                "reco_jet_p4s": [jet_pts[j], jet_etas[j], jet_phis[j], jet_masses[j]],
                "is_tau": is_tau,
                "gen_jet_tau_decaymode": matched_tau["decay_mode"] if matched_tau else -1,
                "gen_jet_tau_charge": matched_tau["charge"] if matched_tau else -999,
                "gen_jet_tau_p4s": [matched_tau["p4"].pt, matched_tau["p4"].eta, matched_tau["p4"].phi, matched_tau["p4"].mass] if matched_tau else [0, 0, 0, 0],
                "reco_cand_deta": [c["cand_deta"] for c in constituents],
                "reco_cand_dphi": [c["cand_dphi"] for c in constituents],
                "reco_cand_logpt": [c["cand_logpt"] for c in constituents],
                "reco_cand_loge": [c["cand_loge"] for c in constituents],
                "reco_cand_logptrel": [c["cand_logptrel"] for c in constituents],
                "reco_cand_logerel": [c["cand_logerel"] for c in constituents],
                "reco_cand_deltaR": [c["cand_deltaR"] for c in constituents],
                "reco_cand_charge": [c["cand_charge"] for c in constituents],
                "reco_cand_isElectron": [c["isElectron"] for c in constituents],
                "reco_cand_isMuon": [c["isMuon"] for c in constituents],
                "reco_cand_isPhoton": [c["isPhoton"] for c in constituents],
                "reco_cand_isChargedHadron": [c["isChargedHadron"] for c in constituents],
                "reco_cand_isNeutralHadron": [c["isNeutralHadron"] for c in constituents],
                "reco_cand_dz": [c["cand_dz"] for c in constituents],
                "reco_cand_dz_error": [c["cand_dz_error"] for c in constituents],
                "reco_cand_dxy": [c["cand_dxy"] for c in constituents],
                "reco_cand_dxy_error": [c["cand_dxy_error"] for c in constituents],
            }
            all_jets.append(jet_record)
            
    out_ak = ak.Array(all_jets)
    ak.to_parquet(out_ak, output_path)
    print(f"Saved {len(all_jets)} jets to {output_path}")

if __name__ == "__main__":
    process_file("ggHtautau-NEVENT10000-RS17000001.parquet", "ggHtautau_jet_level.parquet")

