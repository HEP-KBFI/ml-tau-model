from dataclasses import dataclass

import awkward as ak
import numpy as np
import torch
import tqdm
import vector
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment

from mltau.models.ParTauDETR_module import ParTauDETRModule
from mltau.tools.general import reinitialize_p4
from mltau.tools.io.ParTauDETR_dataloader import ParticleTransformerDETRDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# from hydra import compose, initialize
# from omegaconf import OmegaConf

# with initialize(version_base=None, config_path="../config", job_name="test_app"):
#     cfg = compose(config_name="main_ParTauDETR")


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


def get_predicted_particles(outputs, reco_jet_p4s, obj_cls_trsh: float = 0.5):
    object_probs = torch.softmax(outputs["pred_logits"], dim=-1)
    pred_scores = object_probs[..., 0]
    pred_mask = pred_scores >= obj_cls_trsh

    pred_charge_probs = torch.softmax(outputs["pred_charge_logits"], dim=-1)

    pred_charge_cls = pred_charge_probs.argmax(dim=-1)
    charge_lut = outputs["pred_charge_logits"].new_tensor([-1, 0, 1], dtype=torch.long)
    pred_charge = charge_lut[pred_charge_cls]

    pred_meson_class = outputs["pred_meson_class_logits"].argmax(dim=-1)

    pred_p4 = decode_kinematics(
        kin=outputs["pred_kinematics"],
        reco_pt=reco_jet_p4s["pt"],
        reco_eta=reco_jet_p4s["eta"],
        reco_phi=reco_jet_p4s["phi"],
        reco_energy=reco_jet_p4s["energy"],
    )

    pred_p4 = ak.drop_none(ak.mask(pred_p4, pred_mask))
    pred_charge = ak.drop_none(ak.mask(pred_charge, pred_mask))
    pred_meson_class = ak.drop_none(ak.mask(pred_meson_class, pred_mask))
    return pred_p4, pred_charge, pred_meson_class


def get_true_particles(targets, reco_jet_p4s):
    target_mask = targets["particles_mask"].bool()

    target_charge_cls = targets["particles_charge_ohe"].argmax(dim=-1)
    charge_lut = target_charge_cls.new_tensor([-1, 0, 1], dtype=torch.long)
    target_charge = charge_lut[target_charge_cls]
    target_charge = ak.drop_none(ak.mask(target_charge, target_mask))

    target_meson_class = targets["particles_meson_class_ohe"].argmax(dim=-1)
    target_meson_class = ak.drop_none(ak.mask(target_meson_class, target_mask))

    true_p4 = decode_kinematics(
        kin=targets["particles_kinematics"],
        reco_pt=reco_jet_p4s["pt"],
        reco_eta=reco_jet_p4s["eta"],
        reco_phi=reco_jet_p4s["phi"],
        reco_energy=reco_jet_p4s["energy"],
    )
    true_p4 = ak.drop_none(ak.mask(true_p4, target_mask))
    return true_p4, target_charge, target_meson_class


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


def match_particles(
    pred_p4: ak.Array,
    true_p4: ak.Array,
    pred_charge: ak.Array,
    true_charge: ak.Array,
    pred_meson_class: ak.Array,
    true_meson_class: ak.Array,
    max_dr: float = 0.4,
    mismatch_penalty: float = 5.0,
) -> ak.Array:
    """Match predicted to true particles per event via Hungarian matching.

    Cost = ΔR + penalty * (charge_mismatch + meson_class_mismatch).
    Unmatched particles are excluded.

    Args:
        pred_p4: jagged ak.Array of predicted 4-momenta per event.
        true_p4: jagged ak.Array of true 4-momenta per event.
        pred_charge, true_charge: charge arrays.
        pred_meson_class, true_meson_class: meson class arrays (0 charged, 1 neutral).
        max_dr: maximum ΔR for a valid match.
        mismatch_penalty: additive penalty for charge or PDG mismatch.
            A value of 5.0 means the matcher prefers a correct-identity
            match at ΔR=0.5 over a wrong-identity match at ΔR=0.0.

    Returns:
        ak.Array with fields ``pred_idx``, ``true_idx`` (jagged int per event).
    """
    pred_idx_list = []
    true_idx_list = []

    for evt_p, evt_t, p_ch, t_ch, pred_class, true_class in zip(
        pred_p4,
        true_p4,
        pred_charge,
        true_charge,
        pred_meson_class,
        true_meson_class,
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
        meson_class_mismatch = (
            np.asarray(pred_class)[:, None] != np.asarray(true_class)[None, :]
        ).astype(np.float64)

        cost = dr + mismatch_penalty * (ch_mismatch + meson_class_mismatch)

        pred_idx, true_idx = linear_sum_assignment(cost)

        valid = dr[pred_idx, true_idx] <= max_dr
        pred_idx = pred_idx[valid]
        true_idx = true_idx[valid]

        pred_idx_list.append(ak.Array(pred_idx.astype(np.int64)))
        true_idx_list.append(ak.Array(true_idx.astype(np.int64)))

    return ak.Array(
        {"pred_idx": ak.Array(pred_idx_list), "true_idx": ak.Array(true_idx_list)}
    )


def verbose_true_pred_comparison(
    pred_meson_class: ak.Array,
    target_meson_class: ak.Array,
    pred_charge: ak.Array,
    target_charge: ak.Array,
    pred_p4: ak.Array,
    true_p4: ak.Array,
    data: ak.Array,
    matches: ak.Array,
    n_events: int = 20,
):
    total_pred_p4 = p4_from_components(pred_p4)
    reduced_pred_p4 = p4_from_components(pred_p4[matches.pred_idx])
    total_true_p4 = p4_from_components(true_p4)
    reduced_true_p4 = p4_from_components(true_p4[matches.true_idx])

    for i in range(n_events):
        print("--------------------------------------")
        print("--------------------------------------")
        print(f"------------- Event {i} -----------------")
        print("--------------------------------------")
        print(
            f"Number predicted particles: {len(pred_meson_class[i])}, \t Number true particles: {len(target_meson_class[i])}"
        )
        print("Best matches:")
        print("[Meson class: 0=charged, 1=neutral]")
        print(
            f"Pred: {pred_meson_class[matches.pred_idx][i]}\t True: {target_meson_class[matches.true_idx][i]}"
        )
        print(
            f"AllPred: {pred_meson_class[i]} \t AllTrue: {target_meson_class[i]}"
        )
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


def calculate_metrics(matches, target_meson_class, pred_meson_class):
    n_true = ak.num(target_meson_class).to_numpy()
    n_pred = ak.num(pred_meson_class).to_numpy()
    n_matched = ak.num(matches.pred_idx).to_numpy()
    efficiency = np.divide(
        n_matched, n_true, out=np.zeros_like(n_true, dtype=float), where=n_true != 0
    )
    purity = np.divide(
        n_matched, n_pred, out=np.zeros_like(n_pred, dtype=float), where=n_pred != 0
    )
    num = 2 * (efficiency * purity)
    denom = efficiency + purity
    f1 = np.divide(num, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    return np.mean(f1)


def model_inference(checkpoint_path, data_path, cfg):
    model = ParTauDETRModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=DEVICE,
        cfg=cfg,
    )
    model.to(DEVICE)
    model.eval()
    data = ak.from_parquet(data_path)
    print(f"Read {len(data):,} jets from {data_path}.", flush=True)
    ds = ParticleTransformerDETRDataset.for_arrays(cfg)
    batch = ds.build_tensors(data)

    reco_jet_p4s = batch[6]

    with torch.no_grad():
        outputs, targets, _weights, _, _ = model.forward(batch)
    return outputs, targets, _weights, reco_jet_p4s, data


def create_predictions(
    outputs, targets, _weights, reco_jet_p4s, cfg, obj_cls_trsh=0.885
):
    targets = get_true_particles(targets, reco_jet_p4s)
    predictions = get_predicted_particles(
        outputs,
        reco_jet_p4s,
        obj_cls_trsh=obj_cls_trsh,
    )
    true_daughters = TauDaughter(*targets)
    pred_daughters = TauDaughter(*predictions)
    return true_daughters, pred_daughters


def save_results(true_daughters, pred_daughters, output_path):
    results = ak.Array(
        {
            "pred_p4": pred_daughters.p4,
            "pred_charge": pred_daughters.charge,
            "pred_meson_class": pred_daughters.meson_class,
            "true_p4": true_daughters.p4,
            "true_charge": true_daughters.charge,
            "true_meson_class": true_daughters.meson_class,
        }
    )
    ak.to_parquet(results, output_path)


@dataclass
class TauDaughter:
    p4: torch.Tensor
    charge: torch.Tensor
    meson_class: torch.Tensor


def scan_thresholds(
    outputs,
    reco_jet_p4s,
    true_p4,
    target_charge,
    target_meson_class,
    cfg: DictConfig,
):
    thresholds = np.linspace(0.9, 0.8, 21)
    f1_scores = []
    for obj_cls_trsh in tqdm.tqdm(thresholds):
        pred_p4, pred_charge, pred_meson_class = get_predicted_particles(
            outputs,
            reco_jet_p4s,
            obj_cls_trsh=obj_cls_trsh,
        )
        matches = match_particles(
            pred_p4,
            true_p4,
            pred_charge,
            target_charge,
            pred_meson_class,
            target_meson_class,
            max_dr=0.4,
        )
        f1 = calculate_metrics(matches, target_meson_class, pred_meson_class)
        f1_scores.append(f1)
    best_thrsh_idx = np.argmax(f1_scores)
    best_thrsh = thresholds[best_thrsh_idx]

    pred_p4, pred_charge, pred_meson_class = get_predicted_particles(
        outputs, reco_jet_p4s, obj_cls_trsh=best_thrsh
    )
    matches = match_particles(
        pred_p4,
        true_p4,
        pred_charge,
        target_charge,
        pred_meson_class,
        target_meson_class,
        max_dr=0.4,
    )  # Matches is needed to compare against true particle.
