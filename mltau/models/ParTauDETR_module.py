from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from mltau.models.ParTauDETR import ParTauDETR
from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import TauLoss

try:  # scipy's LAPJVsp solver is ~10x faster than the pure-python fallback below
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    _scipy_lsa = None


def _hungarian_rect_min_cost(cost: list[list[float]]) -> tuple[list[int], list[int]]:
    """Hungarian algorithm for rectangular cost matrices (n_rows <= n_cols)."""
    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows > 0 else 0

    if n_rows == 0 or n_cols == 0:
        return [], []
    if n_rows > n_cols:
        raise ValueError("_hungarian_rect_min_cost expects n_rows <= n_cols.")

    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1

    rows = list(range(n_rows))
    cols = assignment
    return rows, cols


def _solve_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost bipartite matching for one [n_rows, n_cols] cost matrix."""
    if _scipy_lsa is not None:
        return _scipy_lsa(cost)

    n_rows, n_cols = cost.shape
    if n_rows <= n_cols:
        rows, cols = _hungarian_rect_min_cost(cost.tolist())
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    rows_t, cols_t = _hungarian_rect_min_cost(cost.T.tolist())
    pred_idx = np.asarray(cols_t, dtype=np.int64)
    tgt_idx = np.asarray(rows_t, dtype=np.int64)
    order = np.argsort(pred_idx)  # scipy returns row indices in ascending order
    return pred_idx[order], tgt_idx[order]


def _classification_cost_matrix(
    pred_logits: torch.Tensor,
    target_classes: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """
    Per-query/per-target classification matching cost based on -log softmax.

    Args:
        pred_logits: [B, Q, C]
        target_classes: [B, T]
    Returns:
        cost: [B, Q, T] in float32, zero where the target is `ignore_index`.
    """
    # Build matching costs in fp32 for AMP stability.
    nll = -F.log_softmax(pred_logits.float(), dim=-1)  # [B, Q, C]
    valid = target_classes != ignore_index  # [B, T]
    # `gather` needs in-range indices even for the entries we discard afterwards.
    idx = target_classes.clamp_min(0).unsqueeze(1).expand(-1, nll.size(1), -1)
    cost = torch.gather(nll, 2, idx)  # [B, Q, T]
    return cost * valid.unsqueeze(1)


class HungarianMatcher(nn.Module):
    """DETR-style matcher with mixed regression/classification costs."""

    # Cost charged to padded target slots. Constant across queries, so it only
    # shifts the objective by a constant and leaves the optimum over the real
    # targets untouched -- this lets every jet be solved on one fixed [Q, T]
    # matrix instead of a per-jet compacted one.
    _PAD_COST = 1.0e6

    def __init__(
        self,
        cost_objectness: float = 1.0,
        cost_kinematics_l1: float = 2.0,
        cost_charge_ce: float = 1.0,
        cost_pdg_ce: float = 1.0,
        object_class_index: int = 0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.cost_objectness = cost_objectness
        self.cost_kinematics_l1 = cost_kinematics_l1
        self.cost_charge_ce = cost_charge_ce
        self.cost_pdg_ce = cost_pdg_ce
        self.object_class_index = object_class_index
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_kinematics: torch.Tensor,
        pred_charge_logits: torch.Tensor,
        pred_pdg_logits: torch.Tensor,
        target_kinematics: torch.Tensor,
        target_charge_cls: torch.Tensor,
        target_pdg_cls: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_logits: [B, Q, 2]
            pred_kinematics: [B, Q, K]
            pred_charge_logits: [B, Q, C_charge]
            pred_pdg_logits: [B, Q, C_pdg]
            target_kinematics: [B, T, K]
            target_charge_cls: [B, T]
            target_pdg_cls: [B, T]
            target_mask: [B, T]

        Returns:
            (batch_idx, query_idx, target_idx), three flat int64 tensors of equal
            length listing every matched (jet, query, target-slot) triplet.
        """
        device = pred_logits.device
        batch_size, num_queries, _ = pred_logits.shape
        num_targets = target_mask.size(1)

        empty = torch.empty(0, dtype=torch.long, device=device)
        if num_targets == 0 or batch_size == 0:
            return empty, empty, empty

        # ------------------------------------------------------------------
        # All costs are built for the whole batch at once, in fp32 to avoid AMP
        # dtype mismatches. Roughly a dozen kernels per step instead of ~10 per
        # jet, and no host synchronisation until the single transfer below.
        # ------------------------------------------------------------------
        obj_cost = -F.log_softmax(pred_logits.float(), dim=-1)[
            ..., self.object_class_index
        ]  # [B, Q]
        # L1 distance by explicit broadcast: torch.cdist(p=1) has no fast kernel.
        kin_cost = (
            (pred_kinematics.float().unsqueeze(2) - target_kinematics.float().unsqueeze(1))
            .abs()
            .sum(-1)
        )  # [B, Q, T]
        charge_cost = _classification_cost_matrix(
            pred_charge_logits, target_charge_cls, self.ignore_index
        )
        pdg_cost = _classification_cost_matrix(
            pred_pdg_logits, target_pdg_cls, self.ignore_index
        )

        total_cost = (
            self.cost_objectness * obj_cost.unsqueeze(-1)
            + self.cost_kinematics_l1 * kin_cost
            + self.cost_charge_ce * charge_cost
            + self.cost_pdg_ce * pdg_cost
        )
        total_cost = torch.nan_to_num(total_cost, nan=0.0, posinf=1e4, neginf=-1e4)
        total_cost = total_cost.masked_fill(
            ~target_mask.unsqueeze(1), self._PAD_COST
        )

        # One device -> host transfer per step for the whole batch.
        cost_np = total_cost.cpu().numpy()
        valid_np = target_mask.cpu().numpy()
        to_solve = np.flatnonzero(valid_np.any(axis=1))
        if to_solve.size == 0:
            return empty, empty, empty

        # Every solve returns exactly min(Q, T) pairs, so the results pack into
        # a dense array and the padded slots are dropped in one vectorised pass.
        num_pairs = min(num_queries, num_targets)
        query_idx = np.empty((to_solve.size, num_pairs), dtype=np.int64)
        target_idx = np.empty((to_solve.size, num_pairs), dtype=np.int64)
        for i, b in enumerate(to_solve):
            rows, cols = _solve_assignment(cost_np[b])
            query_idx[i] = rows
            target_idx[i] = cols

        batch_idx = np.repeat(to_solve, num_pairs)
        query_idx = query_idx.reshape(-1)
        target_idx = target_idx.reshape(-1)
        keep = valid_np[batch_idx, target_idx]

        return (
            torch.from_numpy(batch_idx[keep]).to(device, non_blocking=True),
            torch.from_numpy(query_idx[keep]).to(device, non_blocking=True),
            torch.from_numpy(target_idx[keep]).to(device, non_blocking=True),
        )


class SetCriterion(nn.Module):
    """DETR-style criterion with objectness + kinematics + charge + pdg
    losses, plus auxiliary consistency penalties."""

    def __init__(
        self,
        matcher: HungarianMatcher,
        tau_loss: TauLoss,
        pdg_class_ids: list[int],
        loss_objectness_weight: float = 1.0,
        loss_tau_id_weight: float = 1.0,
        loss_kinematics_weight: float = 5.0,
        loss_charge_weight: float = 1.0,
        loss_pdg_weight: float = 1.0,
        loss_consistency_weight: float = 0.0,
        loss_charge_count_weight: float = 0.0,
        no_object_class_index: int = 1,
        object_class_index: int = 0,
        eos_coef: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.matcher = matcher
        self.tau_loss = tau_loss
        self.pdg_class_ids = pdg_class_ids
        self.loss_objectness_weight = loss_objectness_weight
        self.loss_tau_id_weight = loss_tau_id_weight
        self.loss_kinematics_weight = loss_kinematics_weight
        self.loss_charge_weight = loss_charge_weight
        self.loss_pdg_weight = loss_pdg_weight
        self.loss_consistency_weight = loss_consistency_weight
        self.loss_charge_count_weight = loss_charge_count_weight
        self.no_object_class_index = no_object_class_index
        self.object_class_index = object_class_index
        self.eos_coef = eos_coef
        self.ignore_index = ignore_index

        # Build (charge_class, pdg_class) validity mask.
        # charge classes: 0 = -1,  1 = 0,  2 = +1
        # pdg classes: indices into pdg_class_ids (abs PDG values)
        n_charge = 3
        n_pdg = len(pdg_class_ids)
        valid = torch.zeros(n_charge, n_pdg, dtype=torch.float32)
        # Neutral-only PDGs can only have charge 0 (class 1)
        neutral_pdgs = {111, 311, 310, 130, 22, 2112, 221, 223}
        # Charged PDGs can have charge ±1 (classes 0 and 2)
        charged_pdgs = {211, 321, 11, 13, 2212, 323}
        for pdg_idx, pdg_abs in enumerate(pdg_class_ids):
            if pdg_abs in charged_pdgs:
                valid[0, pdg_idx] = 1.0  # charge -1
                valid[2, pdg_idx] = 1.0  # charge +1
            elif pdg_abs in neutral_pdgs:
                valid[1, pdg_idx] = 1.0  # charge 0
            # else: unknown → all invalid (no penalty needed)
        self.register_buffer("charge_pdg_valid", valid)  # [3, N_pdg]

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros(())
        if weights is None:
            return values.mean()
        return (values * weights).sum() / (weights.sum() + 1e-8)

    def forward(
        self,
        outputs: dict,
        target_kinematics: torch.Tensor,
        target_charge_cls: torch.Tensor,
        target_pdg_cls: torch.Tensor,
        target_mask: torch.Tensor,
        target_is_tau: torch.Tensor | None = None,
        jet_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_kinematics = outputs["pred_kinematics"]
        pred_charge_logits = outputs["pred_charge_logits"]
        pred_pdg_logits = outputs["pred_pdg_logits"]

        batch_size, num_queries, _ = pred_logits.shape
        device = pred_logits.device

        # Daughter-level losses are signal-only: background jets only supervise
        # the tau-tagging head (loss_tau_id). If no tau label is provided we treat
        # every jet as signal to preserve the previous behaviour.
        if target_is_tau is not None:
            signal_mask = target_is_tau.bool()
        else:
            signal_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Restricting the matcher to signal jets means background jets are never
        # even handed to the assignment solver -- with a 7:1 background:signal mix
        # that alone removes most of the matching work.
        match_mask = target_mask & signal_mask.unsqueeze(1)
        pair_b, pair_q, pair_t = self.matcher(
            pred_logits=pred_logits,
            pred_kinematics=pred_kinematics,
            pred_charge_logits=pred_charge_logits,
            pred_pdg_logits=pred_pdg_logits,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_pdg_cls=target_pdg_cls,
            target_mask=match_mask,
        )
        num_matched = pair_b.numel()

        tgt_classes = torch.full(
            (batch_size, num_queries),
            self.no_object_class_index,
            dtype=torch.long,
            device=device,
        )
        tgt_classes[pair_b, pair_q] = self.object_class_index

        if num_matched > 0:
            if jet_weights is not None:
                pair_w = jet_weights.to(dtype=pred_logits.dtype, device=device)[pair_b]
            else:
                pair_w = pred_logits.new_ones(num_matched)

            kin_pred_cat = pred_kinematics[pair_b, pair_q]
            kin_tgt_cat = target_kinematics[pair_b, pair_t]

            tgt_charge_sel = target_charge_cls[pair_b, pair_t]
            valid_charge = tgt_charge_sel != self.ignore_index
            # cross_entropy zeroes the ignored entries, so folding the validity
            # flag into the weights reproduces the mean over valid pairs only.
            ce_charge = F.cross_entropy(
                pred_charge_logits[pair_b, pair_q],
                tgt_charge_sel,
                reduction="none",
                ignore_index=self.ignore_index,
            )
            charge_w = pair_w * valid_charge.to(pair_w.dtype)

            tgt_pdg_sel = target_pdg_cls[pair_b, pair_t]
            valid_pdg = tgt_pdg_sel != self.ignore_index
            ce_pdg = F.cross_entropy(
                pred_pdg_logits[pair_b, pair_q],
                tgt_pdg_sel,
                reduction="none",
                ignore_index=self.ignore_index,
            )
            pdg_w = pair_w * valid_pdg.to(pair_w.dtype)

        # objectness over all queries
        class_weight = pred_logits.new_tensor([1.0, self.eos_coef])
        ce_per_query = F.cross_entropy(
            pred_logits.transpose(1, 2),
            tgt_classes,
            weight=class_weight,
            reduction="none",
        )
        if jet_weights is not None:
            w = jet_weights.to(dtype=pred_logits.dtype, device=device)
            w = w * signal_mask.to(dtype=w.dtype)
            loss_objectness = (ce_per_query * w[:, None]).sum() / (
                w.sum() * num_queries + 1e-8
            )
        else:
            sig_w = signal_mask.to(dtype=ce_per_query.dtype)
            loss_objectness = (ce_per_query * sig_w[:, None]).sum() / (
                sig_w.sum() * num_queries + 1e-8
            )

        # jet-level tau-tagging loss
        if "is_tau" in outputs and target_is_tau is not None:
            if jet_weights is not None:
                tau_w = jet_weights.to(dtype=pred_logits.dtype, device=device)
            else:
                tau_w = pred_logits.new_ones(batch_size)
            loss_tau_id = self.tau_loss.compute_tagging_loss(
                outputs["is_tau"], target_is_tau, tau_w
            )
        else:
            loss_tau_id = pred_logits.new_zeros(())

        # matched losses
        if num_matched > 0:
            loss_kinematics, kin_components = self.tau_loss.compute_kinematics_loss(
                kin_pred_cat,
                kin_tgt_cat,
                pair_w,
            )
            loss_charge = self._weighted_mean(ce_charge, charge_w)
            loss_pdg = self._weighted_mean(ce_pdg, pdg_w)
            num_charge_supervised = valid_charge.sum()
            num_pdg_supervised = valid_pdg.sum()
        else:
            loss_kinematics = pred_logits.new_zeros(())
            kin_components = {
                "log_pt": pred_logits.new_zeros(()),
                "delta_eta": pred_logits.new_zeros(()),
                "phi_chord": pred_logits.new_zeros(()),
                "log_mass": pred_logits.new_zeros(()),
            }
            loss_charge = pred_logits.new_zeros(())
            loss_pdg = pred_logits.new_zeros(())
            num_charge_supervised = pred_logits.new_zeros(())
            num_pdg_supervised = pred_logits.new_zeros(())

        total_loss = (
            self.loss_objectness_weight * loss_objectness
            + self.loss_tau_id_weight * loss_tau_id
            + self.loss_kinematics_weight * loss_kinematics
            + self.loss_charge_weight * loss_charge
            + self.loss_pdg_weight * loss_pdg
        )

        # ---- Auxiliary penalties ----
        loss_consistency = pred_logits.new_zeros(())
        loss_charge_count = pred_logits.new_zeros(())

        if self.loss_consistency_weight > 0:
            p_charge = F.softmax(pred_charge_logits, dim=-1)  # [B, Q, 3]
            p_pdg = F.softmax(pred_pdg_logits, dim=-1)  # [B, Q, N_pdg]
            p_joint = p_charge[..., :, None] * p_pdg[..., None, :]  # [B, Q, 3, N_pdg]
            invalid_prob = (p_joint * (1 - self.charge_pdg_valid)).sum(dim=(-2, -1))
            sig_w = signal_mask.to(dtype=invalid_prob.dtype)
            loss_consistency = (invalid_prob * sig_w[:, None]).sum() / (
                sig_w.sum() * num_queries + 1e-8
            )
            total_loss = total_loss + self.loss_consistency_weight * loss_consistency

        if self.loss_charge_count_weight > 0:
            p_object = F.softmax(pred_logits, dim=-1)[..., 0]  # [B, Q]
            pred_charge_cls = pred_charge_logits.argmax(dim=-1)  # [B, Q]
            is_charged_pred = (pred_charge_cls != 1).float()  # class 1 = charge 0
            expected_charged = (p_object * is_charged_pred).sum(dim=-1)  # [B]
            # Charged daughters are class 0 (-1) or class 2 (+1); ignore padded slots.
            is_charged_true = (
                (target_charge_cls == 0) | (target_charge_cls == 2)
            ).float()
            n_charged_true = is_charged_true.sum(dim=-1)  # [B]
            excess = F.relu(expected_charged - n_charged_true)
            sig_w = signal_mask.to(dtype=excess.dtype)
            loss_charge_count = (excess * sig_w).sum() / (sig_w.sum() + 1e-8)
            total_loss = total_loss + self.loss_charge_count_weight * loss_charge_count

        return {
            "loss": total_loss,
            "loss_objectness": loss_objectness,
            "loss_tau_id": loss_tau_id,
            "loss_kinematics": loss_kinematics,
            "kinematics_log_pt_loss": kin_components["log_pt"],
            "kinematics_delta_eta_loss": kin_components["delta_eta"],
            "kinematics_phi_chord_loss": kin_components["phi_chord"],
            "kinematics_log_mass_loss": kin_components["log_mass"],
            "loss_charge": loss_charge,
            "loss_pdg": loss_pdg,
            "loss_consistency": loss_consistency,
            "loss_charge_count": loss_charge_count,
            "num_matched": pred_logits.new_tensor(float(num_matched)),
            "num_charge_supervised": num_charge_supervised.to(pred_logits.dtype),
            "num_pdg_supervised": num_pdg_supervised.to(pred_logits.dtype),
        }


# Config subtrees ParTauDETRModule.__init__ reads. `output_dir` is included
# because training.input_scaling.scaler_path interpolates it.
_HPARAM_CFG_KEYS = ("model", "dataset", "training", "output_dir")


class ParTauDETRModule(L.LightningModule):
    """
    Lightning module for ParTauDETR with Hungarian matching and mixed losses.

    Expected target keys from dataloader:
      - particles_kinematics: [B, T, K]
      - particles_charge_ohe: [B, T, 3]
      - particles_pdg_ohe: [B, T, N_PDG]
      - particles_mask: [B, T]
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: fully composed Hydra config. Every architecture choice is read
                from it -- nothing about the network is hardcoded here, so a
                checkpoint's architecture is fully described by its config.
        """
        super().__init__()
        self.cfg = cfg
        self.ignore_index = -100
        # Persist cfg into the checkpoint so `load_from_checkpoint(path)` rebuilds
        # the exact architecture without the caller having to supply a config.
        # Only the subtrees __init__ reads are kept: loggers flatten hparams and
        # resolve every interpolation, so carrying unrelated config (e.g. the
        # metrics plotting tree) turns any unresolvable key elsewhere into a
        # crash at fit() time, and floods the hyperparameter table.
        self.save_hyperparameters(
            {
                "cfg": OmegaConf.masked_copy(
                    cfg, [k for k in _HPARAM_CFG_KEYS if k in cfg]
                )
            }
        )

        arch = cfg.model
        encoder_cfg = arch.encoder
        detr_cfg = arch.detr

        num_charge_classes = int(arch.num_charge_classes)
        if num_charge_classes != 3:
            raise ValueError("This module expects 3 charge classes for {-1, 0, +1}.")

        # The PDG class list is the single source of truth for the head width;
        # deriving it here keeps the model, the dataloader one-hot targets and the
        # decoder LUT from ever disagreeing.
        pdg_class_ids = [int(x) for x in cfg.dataset.tau_daughter_pdg_ids]
        self.pdg_class_ids = pdg_class_ids

        self.tau_loss = TauLoss(
            l_m=float(arch.tau_loss.l_m),
            label_smoothing=float(arch.tau_loss.label_smoothing),
        )
        self.num_kinematics_components = int(arch.num_kinematics_components)

        embed_dims = [int(d) for d in encoder_cfg.embed_dims]
        num_heads = int(encoder_cfg.num_heads)
        decoder_num_heads = detr_cfg.get("decoder_num_heads", None)
        decoder_num_heads = (
            num_heads if decoder_num_heads is None else int(decoder_num_heads)
        )
        embed_dim = embed_dims[-1]
        for label, heads in (("encoder", num_heads), ("decoder", decoder_num_heads)):
            if embed_dim % heads != 0:
                raise ValueError(
                    f"{label} num_heads ({heads}) must divide the model dimension "
                    f"embed_dims[-1] ({embed_dim})."
                )

        self.ParTauDETR = ParTauDETR(
            input_dim=int(cfg.dataset.num_features),
            num_queries=int(arch.num_queries),
            num_charge_classes=num_charge_classes,
            num_pdg_classes=len(pdg_class_ids),
            num_kinematics_components=self.num_kinematics_components,
            # encoder
            num_layers=int(encoder_cfg.num_layers),
            num_heads=num_heads,
            num_cls_layers=int(encoder_cfg.num_cls_layers),
            embed_dims=embed_dims,
            pair_embed_dims=[int(d) for d in encoder_cfg.pair_embed_dims],
            pair_input_dim=int(encoder_cfg.pair_input_dim),
            use_pre_activation_pair=bool(encoder_cfg.use_pre_activation_pair),
            remove_self_pair=bool(encoder_cfg.remove_self_pair),
            activation=str(encoder_cfg.activation),
            metric=str(encoder_cfg.metric),
            trim=bool(encoder_cfg.trim),
            # DETR decoder and heads
            decoder_num_layers=int(detr_cfg.decoder_num_layers),
            decoder_num_heads=decoder_num_heads,
            decoder_ffn_ratio=int(detr_cfg.decoder_ffn_ratio),
            decoder_dropout=float(detr_cfg.decoder_dropout),
            append_global_token=bool(detr_cfg.append_global_token),
            tau_id_head=bool(detr_cfg.tau_id_head),
            head_dropout=float(detr_cfg.head_dropout),
            for_inference=False,
            use_amp=False,
        )

        self.matcher = HungarianMatcher(
            cost_objectness=float(detr_cfg.matcher.cost_objectness),
            cost_kinematics_l1=float(detr_cfg.matcher.cost_kinematics_l1),
            cost_charge_ce=float(detr_cfg.matcher.cost_charge),
            cost_pdg_ce=float(detr_cfg.matcher.cost_pdg_ce),
            object_class_index=0,
            ignore_index=self.ignore_index,
        )

        self.criterion = SetCriterion(
            matcher=self.matcher,
            tau_loss=self.tau_loss,
            pdg_class_ids=pdg_class_ids,
            # Read strictly: a `.get(key, default)` here would silently fall back
            # to a hidden default if the key were renamed or misspelled.
            loss_objectness_weight=float(detr_cfg.loss.weight_objectness),
            loss_tau_id_weight=float(detr_cfg.loss.weight_tau_id),
            loss_kinematics_weight=float(detr_cfg.loss.weight_kinematics),
            loss_charge_weight=float(detr_cfg.loss.weight_charge),
            loss_pdg_weight=float(detr_cfg.loss.weight_pdg),
            loss_consistency_weight=float(detr_cfg.loss.weight_consistency),
            loss_charge_count_weight=float(detr_cfg.loss.weight_charge_count),
            no_object_class_index=1,
            object_class_index=0,
            eos_coef=float(detr_cfg.loss.eos_coef),
            ignore_index=self.ignore_index,
        )

        self.score_threshold = float(detr_cfg.inference.score_threshold)

    @staticmethod
    def _ohe_to_class_indices(
        one_hot: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Convert one-hot [B, T, C] to class indices [B, T].
        All-zero rows map to ignore_index.
        """
        cls = one_hot.argmax(dim=-1)
        has_label = one_hot.sum(dim=-1) > 0
        cls = cls.to(torch.long)
        cls = cls.masked_fill(~has_label, ignore_index)
        return cls

    def _extract_set_targets(
        self, targets: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        required = [
            "particles_kinematics",
            "particles_charge_ohe",
            "particles_pdg_ohe",
            "particles_mask",
        ]
        missing = [k for k in required if k not in targets]
        if len(missing) > 0:
            raise KeyError(
                f"Missing required DETR target keys: {missing}. Available keys: {list(targets.keys())}"
            )

        target_kinematics = targets["particles_kinematics"].float()
        target_mask = targets["particles_mask"].bool()

        if (
            target_kinematics.ndim != 3
            or target_kinematics.size(-1) != self.num_kinematics_components
        ):
            raise ValueError(
                f"Expected particles_kinematics shape [B, T, {self.num_kinematics_components}], "
                f"got {tuple(target_kinematics.shape)}"
            )

        charge_ohe = targets["particles_charge_ohe"].float()
        pdg_ohe = targets["particles_pdg_ohe"].float()

        if charge_ohe.ndim != 3 or charge_ohe.shape[:2] != target_mask.shape:
            raise ValueError(
                f"Expected particles_charge_ohe shape [B, T, C], got {tuple(charge_ohe.shape)}"
            )
        if charge_ohe.size(-1) != 3:
            raise ValueError(
                f"Expected particles_charge_ohe last dim = 3, got {charge_ohe.size(-1)}"
            )

        if pdg_ohe.ndim != 3 or pdg_ohe.shape[:2] != target_mask.shape:
            raise ValueError(
                f"Expected particles_pdg_ohe shape [B, T, C], got {tuple(pdg_ohe.shape)}"
            )
        if pdg_ohe.size(-1) != len(self.pdg_class_ids):
            raise ValueError(
                f"Expected particles_pdg_ohe last dim = {len(self.pdg_class_ids)}, got {pdg_ohe.size(-1)}"
            )

        target_charge_cls = self._ohe_to_class_indices(charge_ohe, self.ignore_index)
        target_pdg_cls = self._ohe_to_class_indices(pdg_ohe, self.ignore_index)

        # Ignore padded slots in class losses/matching costs.
        target_charge_cls = target_charge_cls.masked_fill(
            ~target_mask, self.ignore_index
        )
        target_pdg_cls = target_pdg_cls.masked_fill(~target_mask, self.ignore_index)

        # Jet-level tau-tagging label. Prefer an explicit `is_tau` target when the
        # dataloader provides it; otherwise fall back to "has at least one valid
        # daughter", which is the same signal/background distinction.
        if "is_tau" in targets:
            target_is_tau = targets["is_tau"].long()
        else:
            target_is_tau = target_mask.any(dim=1).long()
        if target_is_tau.ndim != 1 or target_is_tau.size(0) != target_mask.size(0):
            raise ValueError(
                f"Expected is_tau shape [{target_mask.size(0)}], got {tuple(target_is_tau.shape)}"
            )

        return (
            target_kinematics,
            target_charge_cls,
            target_pdg_cls,
            target_mask,
            target_is_tau,
        )

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        outputs = self.ParTauDETR(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        return outputs, inputs.target, inputs.weight

    def training_step(self, batch, _batch_idx):
        outputs, targets, weights = self.forward(batch)
        (
            target_kinematics,
            target_charge_cls,
            target_pdg_cls,
            target_mask,
            target_is_tau,
        ) = self._extract_set_targets(targets)

        losses = self.criterion(
            outputs=outputs,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_pdg_cls=target_pdg_cls,
            target_mask=target_mask,
            target_is_tau=target_is_tau,
            jet_weights=weights,
        )

        self.log("train_losses/loss", losses["loss"], on_step=False, on_epoch=True)
        self.log(
            "train_losses/objectness",
            losses["loss_objectness"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/tau_id",
            losses["loss_tau_id"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics",
            losses["loss_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_log_pt_loss",
            losses["kinematics_log_pt_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_delta_eta_loss",
            losses["kinematics_delta_eta_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_phi_chord_loss",
            losses["kinematics_phi_chord_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_log_mass_loss",
            losses["kinematics_log_mass_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/charge", losses["loss_charge"], on_step=False, on_epoch=True
        )
        self.log(
            "train_losses/pdg_loss", losses["loss_pdg"], on_step=False, on_epoch=True
        )
        self.log(
            "train_losses/consistency",
            losses["loss_consistency"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/charge_count",
            losses["loss_charge_count"],
            on_step=False,
            on_epoch=True,
        )

        return losses["loss"]

    def validation_step(self, batch, _batch_idx):
        outputs, targets, weights = self.forward(batch)
        (
            target_kinematics,
            target_charge_cls,
            target_pdg_cls,
            target_mask,
            target_is_tau,
        ) = self._extract_set_targets(targets)

        losses = self.criterion(
            outputs=outputs,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_pdg_cls=target_pdg_cls,
            target_mask=target_mask,
            target_is_tau=target_is_tau,
            jet_weights=weights,
        )

        self.log("val_losses/loss", losses["loss"], on_step=False, on_epoch=True)
        self.log(
            "val_losses/objectness",
            losses["loss_objectness"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/tau_id",
            losses["loss_tau_id"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics",
            losses["loss_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_log_pt_loss",
            losses["kinematics_log_pt_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_delta_eta_loss",
            losses["kinematics_delta_eta_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_phi_chord_loss",
            losses["kinematics_phi_chord_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_log_mass_loss",
            losses["kinematics_log_mass_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/charge", losses["loss_charge"], on_step=False, on_epoch=True
        )
        self.log(
            "val_losses/pdg_loss", losses["loss_pdg"], on_step=False, on_epoch=True
        )
        self.log(
            "val_losses/consistency",
            losses["loss_consistency"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/charge_count",
            losses["loss_charge_count"],
            on_step=False,
            on_epoch=True,
        )

        return losses["loss"]

    def predict_step(self, batch, _batch_idx):
        outputs, _, _ = self.forward(batch)

        object_scores = torch.softmax(outputs["pred_logits"], dim=-1)[..., 0]
        pred_mask = object_scores > self.score_threshold

        charge_class = outputs["pred_charge_logits"].argmax(dim=-1)
        charge_value_lut = outputs["pred_charge_logits"].new_tensor(
            [-1, 0, 1], dtype=torch.long
        )
        pred_charge = charge_value_lut[charge_class]

        pdg_class = outputs["pred_pdg_logits"].argmax(dim=-1)
        pdg_lut = torch.tensor(
            self.pdg_class_ids, dtype=torch.long, device=pdg_class.device
        )
        pred_pdg = pdg_lut[pdg_class]

        result = {
            "pred_kinematics": outputs["pred_kinematics"],
            "pred_charge_logits": outputs["pred_charge_logits"],
            "pred_pdg_logits": outputs["pred_pdg_logits"],
            "pred_logits": outputs["pred_logits"],
            "pred_scores": object_scores,
            "pred_mask": pred_mask,
            "pred_charge": pred_charge,
            "pred_pdg": pred_pdg,
        }

        if "is_tau" in outputs:
            result["is_tau_logits"] = outputs["is_tau"]
            result["is_tau"] = torch.softmax(outputs["is_tau"], dim=-1)[..., 1]

        return result

    def test_step(self, batch, _batch_idx):
        return self.predict_step(batch, _batch_idx)

    def configure_optimizers(self) -> Any:
        base_lr = self.cfg.training.lr
        optimizer = torch.optim.AdamW(
            self.ParTauDETR.parameters(), lr=base_lr, weight_decay=1e-2
        )

        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if estimated_steps is None or estimated_steps <= 0:
            max_epochs = self.cfg.training.trainer.max_epochs
            estimated_steps_per_epoch = 500
            total_steps = max_epochs * estimated_steps_per_epoch
            print(
                f"Warning: Using estimated total_steps={total_steps} (estimated_stepping_batches not available)"
            )
        else:
            total_steps = estimated_steps
            print(
                f"Using calculated total_steps={total_steps} from estimated_stepping_batches"
            )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
