import awkward as ak
import numpy as np
import torch
import vector
from scipy.optimize import linear_sum_assignment

from mltau.models.ParTauDETR_module import ParTauDETRModule
from mltau.tools.io.ParTauDETR_dataloader import ParticleTransformerDETRDataset

# from mltau.tools.general import reinitialize_p4

# from hydra import compose, initialize
# from omegaconf import OmegaConf

# with initialize(version_base=None, config_path="../config", job_name="test_app"):
#     cfg = compose(config_name="main_ParTauDETR")


# checkpoint_path = "/home/laurits/Projects/ml-tau/ml-tau-model/outputs/parataudetr/models/ParTauDETR-model_best.ckpt"
# PARQUET_PATH = "/home/laurits/tmp/ParTauDETR_dataset/z_test.parquet"


# num_queries: int = 16
# num_charge_classes: int = 3
# num_pdg_classes: int = 9
# num_kinematics_components: int = 5

# PDG_CLASS_IDS = [
#         211,
#         111,
#         321,
#         311,
#         310,
#         130,
#         11,
#         13,
#         22,
#         2212,
#         2112,
#         221,
#         323,
#         223,
# ]
# pdg_class_ids = [int(x) for x in PDG_CLASS_IDS]

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_ParTauDETR(data_path, checkpoint_path, cfg):
    model = ParTauDETRModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
        cfg=cfg,
        input_dim=17,
        num_queries=cfg.num_queries,
        pdg_class_ids=cfg.pdg_class_ids,
        num_charge_classes=cfg.num_charge_classes,
    )
    model.to(device)
    model.eval()
    data = ak.from_parquet(data_path)

    ds = ParticleTransformerDETRDataset(row_groups=[], cfg=cfg, batch_size=1)
    batch = ds.build_tensors(data)

    reco_jet_p4s = batch[6]

    with torch.no_grad():
        outputs, targets, _weights = model.forward(batch)

    true_p4, target_charge, target_pdg = get_true_particles(
        targets, reco_jet_p4s, cfg.pdg_class_ids
    )

    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for obj_cls_trsh in thresholds:
        pred_p4, pred_charge, pred_pdg = get_predicted_particles(
            outputs,
            reco_jet_p4s,
            cfg.dataset.tau_daughter_pdg_ids,
            obj_cls_trsh=obj_cls_trsh,
        )
        matches = match_particles(
            pred_p4,
            true_p4,
            pred_charge,
            target_charge,
            pred_pdg,
            target_pdg,
            max_dr=0.4,
        )
        _, _, f1 = calculate_metrics(matches, target_pdg, pred_pdg)
        f1_scores.append(f1)
    best_thrsh_idx = np.argmax(f1_scores)
    best_thrsh = thresholds[best_thrsh_idx]

    pred_p4, pred_charge, pred_pdg = get_predicted_particles(
        outputs, reco_jet_p4s, cfg.dataset.tau_daughter_pdg_ids, obj_cls_trsh=best_thrsh
    )
    matches = match_particles(
        pred_p4,
        true_p4,
        pred_charge,
        target_charge,
        pred_pdg,
        target_pdg,
        max_dr=0.4,
    )

    compare_true_pred(
        pred_pdg,
        target_pdg,
        pred_charge,
        target_charge,
        pred_p4,
        true_p4,
        data,
        matches,
    )


def calculate_metrics(matches, target_pdg, pred_pdg):
    n_true = ak.num(target_pdg)
    n_pred = ak.num(pred_pdg)
    n_matched = ak.num(matches.pred_idx)
    efficiency = n_matched / n_true
    purity = n_matched / n_pred
    f1 = 2 * (efficiency * purity) / (efficiency + purity)
    return efficiency, purity, f1


def get_predicted_particles(
    outputs, reco_jet_p4s, pdg_class_ids, obj_cls_trsh: float = 0.5
):
    """Actual separation happens closer to obj_cls_trsh=[0.8-0.9] for well-trained models"""
    object_probs = torch.softmax(outputs["pred_logits"], dim=-1)
    pred_scores = object_probs[..., 0]
    pred_mask = pred_scores >= obj_cls_trsh

    pred_charge_probs = torch.softmax(outputs["pred_charge_logits"], dim=-1)
    pred_pdg_probs = torch.softmax(outputs["pred_pdg_logits"], dim=-1)

    pred_charge_cls = pred_charge_probs.argmax(dim=-1)
    charge_lut = outputs["pred_charge_logits"].new_tensor([-1, 0, 1], dtype=torch.long)
    pred_charge = charge_lut[pred_charge_cls]

    pred_pdg_cls = pred_pdg_probs.argmax(dim=-1)
    pdg_lut = torch.tensor(pdg_class_ids, dtype=torch.long, device=device)
    pred_pdg = pdg_lut[pred_pdg_cls]

    pred_p4 = decode_kinematics(
        kin=outputs["pred_kinematics"],
        reco_pt=reco_jet_p4s["pt"],
        reco_eta=reco_jet_p4s["eta"],
        reco_phi=reco_jet_p4s["phi"],
        reco_energy=reco_jet_p4s["energy"],
    )

    pred_p4 = ak.drop_none(ak.mask(pred_p4, pred_mask))
    pred_charge = ak.drop_none(ak.mask(pred_charge, pred_mask))
    pred_pdg = ak.drop_none(ak.mask(pred_pdg, pred_mask))
    return pred_p4, pred_charge, pred_pdg


def get_true_particles(targets, reco_jet_p4s, pdg_class_ids):
    target_mask = targets["particles_mask"].bool()

    target_charge_cls = targets["particles_charge_ohe"].argmax(dim=-1)
    charge_lut = target_charge_cls.new_tensor([-1, 0, 1], dtype=torch.long)
    target_charge = charge_lut[target_charge_cls]
    target_charge = ak.drop_none(ak.mask(target_charge, target_mask))

    target_pdg_cls = targets["particles_pdg_ohe"].argmax(dim=-1)
    pdg_lut = torch.tensor(pdg_class_ids, dtype=torch.long, device=device)
    target_pdg = pdg_lut[target_pdg_cls]
    target_pdg = ak.drop_none(ak.mask(target_pdg, target_mask))

    true_p4 = decode_kinematics(
        kin=targets["particles_kinematics"],
        reco_pt=reco_jet_p4s["pt"],
        reco_eta=reco_jet_p4s["eta"],
        reco_phi=reco_jet_p4s["phi"],
        reco_energy=reco_jet_p4s["energy"],
    )
    true_p4 = ak.drop_none(ak.mask(true_p4, target_mask))
    return true_p4, target_charge, target_pdg


def match_particles(
    pred_p4: ak.Array,
    true_p4: ak.Array,
    pred_charge: ak.Array,
    true_charge: ak.Array,
    pred_pdg: ak.Array,
    true_pdg: ak.Array,
    max_dr: float = 0.4,
    mismatch_penalty: float = 5.0,
) -> ak.Array:
    """Match predicted to true particles per event via Hungarian matching.

    Cost = ΔR + penalty * (charge_mismatch + pdg_mismatch).
    Unmatched particles are excluded.


    Args:
        pred_p4: jagged ak.Array of predicted 4-momenta per event.
        true_p4: jagged ak.Array of true 4-momenta per event.
        pred_charge, true_charge: charge arrays.
        pred_pdg, true_pdg: PDG ID arrays.
        max_dr: maximum ΔR for a valid match.
        mismatch_penalty: additive penalty for charge or PDG mismatch.
            A value of 5.0 means the matcher prefers a correct-identity
            match at ΔR=0.5 over a wrong-identity match at ΔR=0.0.

    Returns:
        ak.Array with fields ``pred_idx``, ``true_idx`` (jagged int per event).
    """
    pred_idx_list = []
    true_idx_list = []

    for evt_p, evt_t, p_ch, t_ch, p_pdg, t_pdg in zip(
        pred_p4, true_p4, pred_charge, true_charge, pred_pdg, true_pdg
    ):
        n_pred = len(evt_p)
        n_true = len(evt_t)

        # if n_pred == 0 or n_true == 0:
        #     pred_idx_list.append(ak.Array([], dtype=np.int64))
        #     true_idx_list.append(ak.Array([], dtype=np.int64))
        #     continue

        pred_eta = np.asarray(evt_p.eta)
        pred_phi = np.asarray(evt_p.phi)
        true_eta = np.asarray(evt_t.eta)
        true_phi = np.asarray(evt_t.phi)

        deta = np.abs(pred_eta[:, None] - true_eta[None, :])
        dphi = np.abs(pred_phi[:, None] - true_phi[None, :])
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dr = np.sqrt(deta**2 + dphi**2)  # [n_pred, n_true]

        ch_mismatch = (np.asarray(p_ch)[:, None] != np.asarray(t_ch)[None, :]).astype(
            np.float64
        )
        pdg_mismatch = (
            np.asarray(p_pdg)[:, None] != np.asarray(t_pdg)[None, :]
        ).astype(np.float64)

        cost = dr + mismatch_penalty * (ch_mismatch + pdg_mismatch)

        pred_idx, true_idx = linear_sum_assignment(cost)

        valid = dr[pred_idx, true_idx] <= max_dr
        pred_idx = pred_idx[valid]
        true_idx = true_idx[valid]

        pred_idx_list.append(ak.Array(pred_idx.astype(np.int64)))
        true_idx_list.append(ak.Array(true_idx.astype(np.int64)))

    return ak.Array(
        {"pred_idx": ak.Array(pred_idx_list), "true_idx": ak.Array(true_idx_list)}
    )


def decode_kinematics(
    kin: torch.Tensor,
    reco_pt: torch.Tensor,
    reco_eta: torch.Tensor,
    reco_phi: torch.Tensor,
    reco_energy: torch.Tensor,
) -> ak.Array:
    """
    Convert 5D kinematics target/predictions back to easy to understand p4.

    Input kin order:
      [log(pt_dau/pt_jet), delta_eta, sin(delta_phi), cos(delta_phi), log(m_dau/m_jet)]
    """
    eps = 1e-6

    pt_jet = reco_pt[:, None]
    eta_jet = reco_eta[:, None]
    phi_jet = reco_phi[:, None]

    mass_jet = torch.sqrt(
        torch.clamp(reco_energy**2 - (reco_pt * torch.cosh(reco_eta)) ** 2, min=0.0)
    )
    mass_jet = torch.clamp(mass_jet, min=eps)[:, None]

    log_pt_ratio = kin[..., 0]
    delta_eta = kin[..., 1]
    sin_dphi = kin[..., 2]
    cos_dphi = kin[..., 3]
    log_mass_ratio = kin[..., 4]

    pt = torch.exp(log_pt_ratio) * pt_jet
    eta = delta_eta + eta_jet
    dphi = torch.atan2(sin_dphi, cos_dphi)

    phi = phi_jet + dphi
    phi = torch.atan2(torch.sin(phi), torch.cos(phi))

    mass = torch.exp(log_mass_ratio) * mass_jet

    pred_p4 = vector.awk(
        ak.zip(
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "mass": mass,
            }
        )
    )
    return pred_p4


def p4_from_components(p4):
    total = vector.awk(
        ak.zip(
            {
                "px": ak.sum(p4.px, axis=1),
                "py": ak.sum(p4.py, axis=1),
                "pz": ak.sum(p4.pz, axis=1),
                "energy": ak.sum(p4.energy, axis=1),
            }
        )
    )
    return total


def compare_true_pred(
    pred_pdg: ak.Array,
    target_pdg: ak.Array,
    pred_charge: ak.Array,
    target_charge: ak.Array,
    pred_p4: ak.Array,
    true_p4: ak.Array,
    data: ak.Array,
    matches: ak.Array,
):
    total_pred_p4 = p4_from_components(pred_p4)
    reduced_pred_p4 = p4_from_components(pred_p4[matches.pred_idx])
    total_true_p4 = p4_from_components(true_p4)
    reduced_true_p4 = p4_from_components(true_p4[matches.true_idx])

    for i in range(20):
        print("--------------------------------------")
        print("--------------------------------------")
        print(f"------------- Event {i} -----------------")
        print("--------------------------------------")
        print(
            f"Number predicted particles: {len(pred_pdg[i])}, \t Number true particles: {len(target_pdg[i])}"
        )
        print("Best matches:")
        print("[PDG]")
        print(
            f"Pred: {pred_pdg[matches.pred_idx][i]}\t True: {target_pdg[matches.true_idx][i]}"
        )
        print(f"AllPred: {pred_pdg[i]} \t AllTrue: {target_pdg[i]}")
        print("[Ch]")
        print(
            f"Pred: {pred_charge[matches.pred_idx][i]}\t True: {target_charge[matches.true_idx][i]}"
        )
        print(f"AllPred: {pred_charge[i]} \t AllTrue: {target_charge[i]}")
        print("[pT]")
        print(
            f"Pred: {pred_p4.pt[matches.pred_idx][i]}\t True: {true_p4.pt[matches.true_idx][i]}"
        )
        print(f"AllPred: {pred_p4.pt[i]} \t AllTrue: {true_p4.pt[i]}")
        print()
        print(f"RecoJet constituent PDGs: {data.reco_cand_pdgs[i]}")
        print(f"RecoJet constituent pTs: {reinitialize_p4(data.reco_cand_p4s).pt[i]}")
        print("--------------------------------------")
        print(
            r"Pred $\tau p_T$ (all predicted): ",
            total_pred_p4.pt[i],
            r"$\tau p_T$ (matched): ",
            reduced_pred_p4.pt[i],
        )
        print(
            r"True $\tau p_T$ (all true): ",
            total_true_p4.pt[i],
            r"$\tau p_T$ (matched): ",
            reduced_true_p4.pt[i],
        )
        print()
        print(
            r"Pred $\tau \phi$ (all predicted): ",
            total_pred_p4.phi[i],
            r"$\tau \phi$ (matched): ",
            reduced_pred_p4.phi[i],
        )
        print(
            r"True $\tau \phi$ (all true): ",
            total_true_p4.phi[i],
            r"$\tau \phi$ (matched): ",
            reduced_true_p4.phi[i],
        )
        print()
        print(
            r"Pred $\tau \eta$ (all predicted): ",
            total_pred_p4.eta[i],
            r"$\tau \eta$ (matched): ",
            reduced_pred_p4.eta[i],
        )
        print(
            r"True $\tau \eta$ (all true): ",
            total_true_p4.eta[i],
            r"$\tau \eta$ (matched): ",
            reduced_true_p4.eta[i],
        )
        print()
        print(
            r"Pred $\tau mass$ (all predicted): ",
            total_pred_p4.mass[i],
            r"$\tau mass$ (matched): ",
            reduced_pred_p4.mass[i],
        )
        print(
            r"True $\tau mass$ (all true): ",
            total_true_p4.mass[i],
            r"$\tau mass$ (matched): ",
            reduced_true_p4.mass[i],
        )
