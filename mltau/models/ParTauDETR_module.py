from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from mltau.models.ParTauDETR import ParTauDETR
from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import TauLoss


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


def hungarian_min_cost_assignment(
    cost_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum-cost bipartite matching for a single cost matrix [N_pred, N_tgt]."""
    n_pred, n_tgt = cost_matrix.shape
    device = cost_matrix.device

    if n_pred == 0 or n_tgt == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    if n_pred <= n_tgt:
        rows, cols = _hungarian_rect_min_cost(cost_matrix.detach().cpu().tolist())
        pred_idx = torch.tensor(rows, dtype=torch.long, device=device)
        tgt_idx = torch.tensor(cols, dtype=torch.long, device=device)
        return pred_idx, tgt_idx

    rows_t, cols_t = _hungarian_rect_min_cost(cost_matrix.t().detach().cpu().tolist())
    pred_idx = torch.tensor(cols_t, dtype=torch.long, device=device)
    tgt_idx = torch.tensor(rows_t, dtype=torch.long, device=device)
    return pred_idx, tgt_idx


def _classification_cost_matrix(
    pred_logits: torch.Tensor,
    target_classes: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """
    Per-query/per-target classification matching cost based on -log softmax.

    Args:
        pred_logits: [Q, C]
        target_classes: [T]
    Returns:
        cost: [Q, T] in float32
    """
    # Build matching costs in fp32 for AMP stability.
    nll = -F.log_softmax(pred_logits.float(), dim=-1)
    q = pred_logits.size(0)
    t = target_classes.numel()
    cost = torch.zeros((q, t), dtype=nll.dtype, device=pred_logits.device)

    valid = target_classes != ignore_index
    if valid.any():
        idx = target_classes[valid].to(torch.long)
        cost[:, valid] = nll[:, idx]
    return cost


class HungarianMatcher(nn.Module):
    """DETR-style matcher with mixed regression/classification costs."""

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
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
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
        """
        batch_size = pred_logits.size(0)
        assignments: list[tuple[torch.Tensor, torch.Tensor]] = []

        for b in range(batch_size):
            valid_tgt = target_mask[b]
            tgt_kin = target_kinematics[b][valid_tgt]
            tgt_charge = target_charge_cls[b][valid_tgt]
            tgt_pdg = target_pdg_cls[b][valid_tgt]

            if tgt_kin.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_logits.device)
                assignments.append((empty, empty))
                continue

            # Build all matching costs in fp32 to avoid AMP dtype mismatches.
            obj_cost = -F.log_softmax(pred_logits[b].float(), dim=-1)[
                :, self.object_class_index
            ]
            kin_cost = torch.cdist(pred_kinematics[b].float(), tgt_kin.float(), p=1)
            charge_cost = _classification_cost_matrix(
                pred_charge_logits[b], tgt_charge, self.ignore_index
            )
            pdg_cost = _classification_cost_matrix(
                pred_pdg_logits[b], tgt_pdg, self.ignore_index
            )

            total_cost = (
                self.cost_objectness * obj_cost[:, None]
                + self.cost_kinematics_l1 * kin_cost
                + self.cost_charge_ce * charge_cost
                + self.cost_pdg_ce * pdg_cost
            )

            pred_idx, tgt_idx = hungarian_min_cost_assignment(total_cost)
            assignments.append((pred_idx, tgt_idx))

        return assignments


class SetCriterion(nn.Module):
    """DETR-style criterion with objectness + kinematics + charge + pdg losses."""

    def __init__(
        self,
        matcher: HungarianMatcher,
        tau_loss: TauLoss,
        loss_objectness_weight: float = 1.0,
        loss_kinematics_weight: float = 5.0,
        loss_charge_weight: float = 1.0,
        loss_pdg_weight: float = 1.0,
        no_object_class_index: int = 1,
        object_class_index: int = 0,
        eos_coef: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.matcher = matcher
        self.tau_loss = tau_loss
        self.loss_objectness_weight = loss_objectness_weight
        self.loss_kinematics_weight = loss_kinematics_weight
        self.loss_charge_weight = loss_charge_weight
        self.loss_pdg_weight = loss_pdg_weight
        self.no_object_class_index = no_object_class_index
        self.object_class_index = object_class_index
        self.eos_coef = eos_coef
        self.ignore_index = ignore_index

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
        jet_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_kinematics = outputs["pred_kinematics"]
        pred_charge_logits = outputs["pred_charge_logits"]
        pred_pdg_logits = outputs["pred_pdg_logits"]

        batch_size, num_queries, _ = pred_logits.shape
        device = pred_logits.device

        assignments = self.matcher(
            pred_logits=pred_logits,
            pred_kinematics=pred_kinematics,
            pred_charge_logits=pred_charge_logits,
            pred_pdg_logits=pred_pdg_logits,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_pdg_cls=target_pdg_cls,
            target_mask=target_mask,
        )

        tgt_classes = torch.full(
            (batch_size, num_queries),
            self.no_object_class_index,
            dtype=torch.long,
            device=device,
        )

        kin_pred = []
        kin_tgt = []
        kin_weights = []
        charge_losses = []
        charge_weights = []
        pdg_losses = []
        pdg_weights = []

        for b, (pred_idx, tgt_idx) in enumerate(assignments):
            if pred_idx.numel() == 0:
                continue

            tgt_classes[b, pred_idx] = self.object_class_index

            valid_tgt = target_mask[b]
            tgt_kin_valid = target_kinematics[b][valid_tgt]
            tgt_charge_valid = target_charge_cls[b][valid_tgt]
            tgt_pdg_valid = target_pdg_cls[b][valid_tgt]

            pred_kin_sel = pred_kinematics[b, pred_idx]
            tgt_kin_sel = tgt_kin_valid[tgt_idx]
            kin_pred.append(pred_kin_sel)
            kin_tgt.append(tgt_kin_sel)

            if jet_weights is not None:
                pair_w = (
                    jet_weights[b].to(dtype=pred_logits.dtype).expand(pred_idx.numel())
                )
            else:
                pair_w = pred_logits.new_ones(pred_idx.numel())
            kin_weights.append(pair_w)

            # charge CE on matched pairs with valid labels
            tgt_charge_sel = tgt_charge_valid[tgt_idx]
            pred_charge_sel = pred_charge_logits[b, pred_idx]
            valid_charge = tgt_charge_sel != self.ignore_index
            if valid_charge.any():
                ce_charge = F.cross_entropy(
                    pred_charge_sel[valid_charge],
                    tgt_charge_sel[valid_charge],
                    reduction="none",
                    ignore_index=self.ignore_index,
                )
                charge_losses.append(ce_charge)
                charge_weights.append(pair_w[valid_charge])

            # pdg CE on matched pairs with valid labels
            tgt_pdg_sel = tgt_pdg_valid[tgt_idx]
            pred_pdg_sel = pred_pdg_logits[b, pred_idx]
            valid_pdg = tgt_pdg_sel != self.ignore_index
            if valid_pdg.any():
                ce_pdg = F.cross_entropy(
                    pred_pdg_sel[valid_pdg],
                    tgt_pdg_sel[valid_pdg],
                    reduction="none",
                    ignore_index=self.ignore_index,
                )
                pdg_losses.append(ce_pdg)
                pdg_weights.append(pair_w[valid_pdg])

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
            loss_objectness = (ce_per_query * w[:, None]).sum() / (
                w.sum() * num_queries + 1e-8
            )
        else:
            loss_objectness = ce_per_query.mean()

        # matched losses
        if len(kin_pred) > 0:
            kin_pred_cat = torch.cat(kin_pred, dim=0)
            kin_tgt_cat = torch.cat(kin_tgt, dim=0)
            kin_w = torch.cat(kin_weights, dim=0)
            loss_kinematics, kin_components = self.tau_loss.compute_kinematics_loss(
                kin_pred_cat,
                kin_tgt_cat,
                kin_w,
            )
        else:
            loss_kinematics = pred_logits.new_zeros(())
            kin_components = {
                "log_pt": pred_logits.new_zeros(()),
                "delta_eta": pred_logits.new_zeros(()),
                "phi_chord": pred_logits.new_zeros(()),
                "log_mass": pred_logits.new_zeros(()),
            }

        if len(charge_losses) > 0:
            charge_vals = torch.cat(charge_losses, dim=0)
            charge_w = torch.cat(charge_weights, dim=0)
            loss_charge = self._weighted_mean(charge_vals, charge_w)
            num_charge_supervised = charge_vals.numel()
        else:
            loss_charge = pred_logits.new_zeros(())
            num_charge_supervised = 0

        if len(pdg_losses) > 0:
            pdg_vals = torch.cat(pdg_losses, dim=0)
            pdg_w = torch.cat(pdg_weights, dim=0)
            loss_pdg = self._weighted_mean(pdg_vals, pdg_w)
            num_pdg_supervised = pdg_vals.numel()
        else:
            loss_pdg = pred_logits.new_zeros(())
            num_pdg_supervised = 0

        total_loss = (
            self.loss_objectness_weight * loss_objectness
            + self.loss_kinematics_weight * loss_kinematics
            + self.loss_charge_weight * loss_charge
            + self.loss_pdg_weight * loss_pdg
        )

        num_matched = sum(int(pred_idx.numel()) for pred_idx, _ in assignments)

        return {
            "loss": total_loss,
            "loss_objectness": loss_objectness,
            "loss_kinematics": loss_kinematics,
            "kinematics_log_pt_loss": kin_components["log_pt"],
            "kinematics_delta_eta_loss": kin_components["delta_eta"],
            "kinematics_phi_chord_loss": kin_components["phi_chord"],
            "kinematics_log_mass_loss": kin_components["log_mass"],
            "loss_charge": loss_charge,
            "loss_pdg": loss_pdg,
            "num_matched": pred_logits.new_tensor(float(num_matched)),
            "num_charge_supervised": pred_logits.new_tensor(
                float(num_charge_supervised)
            ),
            "num_pdg_supervised": pred_logits.new_tensor(float(num_pdg_supervised)),
        }


class ParTauDETRModule(L.LightningModule):
    """
    Lightning module for ParTauDETR with Hungarian matching and mixed losses.

    Expected target keys from dataloader:
      - particles_kinematics: [B, T, K]
      - particles_charge_ohe: [B, T, 3]
      - particles_pdg_ohe: [B, T, N_PDG]
      - particles_mask: [B, T]
    """

    def __init__(
        self,
        cfg: DictConfig,
        input_dim: int,
        num_queries: int = 8,
        num_charge_classes: int = 3,
    ):
        super().__init__()
        self.cfg = cfg
        self.ignore_index = -100

        if num_charge_classes != 3:
            raise ValueError("This module expects 3 charge classes for {-1, 0, +1}.")

        pdg_class_ids = [int(x) for x in cfg.dataset.tau_daughter_pdg_ids]

        self.pdg_class_ids = pdg_class_ids
        self.tau_loss = TauLoss(
            l_m=0.2, label_smoothing=0.1
        )  # TODO: since the masses are rather well reconstructed, I guess the penalty does not need to be so big.
        self.num_kinematics_components = cfg.model.num_kinematics_components

        self.ParTauDETR = ParTauDETR(
            input_dim=input_dim,
            num_queries=num_queries,
            num_charge_classes=num_charge_classes,
            num_pdg_classes=len(pdg_class_ids),
            num_kinematics_components=self.num_kinematics_components,
            num_layers=2,
            embed_dims=[256, 512, 256],
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="theta-phi",
            append_global_token=True,
        )

        self.matcher = HungarianMatcher(
            cost_objectness=cfg.model.detr.matcher.cost_objectness,
            cost_kinematics_l1=cfg.model.detr.matcher.cost_kinematics_l1,
            cost_charge_ce=cfg.model.detr.matcher.cost_charge,
            cost_pdg_ce=cfg.model.detr.matcher.cost_pdg_ce,
            object_class_index=0,
            ignore_index=self.ignore_index,
        )

        self.criterion = SetCriterion(
            matcher=self.matcher,
            tau_loss=self.tau_loss,
            loss_objectness_weight=cfg.model.detr.loss.weight_objectness,
            loss_kinematics_weight=cfg.model.detr.loss.weight_kinematics,
            loss_charge_weight=cfg.model.detr.loss.weight_charge,
            loss_pdg_weight=cfg.model.detr.loss.weight_pdg,
            no_object_class_index=1,
            object_class_index=0,
            eos_coef=cfg.model.detr.loss.eos_coef,
            ignore_index=self.ignore_index,
        )

        self.score_threshold = cfg.model.detr.inference.score_threshold

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return target_kinematics, target_charge_cls, target_pdg_cls, target_mask

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
        target_kinematics, target_charge_cls, target_pdg_cls, target_mask = (
            self._extract_set_targets(targets)
        )

        losses = self.criterion(
            outputs=outputs,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_pdg_cls=target_pdg_cls,
            target_mask=target_mask,
            jet_weights=weights,
        )

        self.log("train_losses/loss", losses["loss"], on_step=True, on_epoch=True)
        self.log(
            "train_losses/objectness",
            losses["loss_objectness"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics",
            losses["loss_kinematics"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_log_pt_loss",
            losses["kinematics_log_pt_loss"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_delta_eta_loss",
            losses["kinematics_delta_eta_loss"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_phi_chord_loss",
            losses["kinematics_phi_chord_loss"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_log_mass_loss",
            losses["kinematics_log_mass_loss"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_losses/charge", losses["loss_charge"], on_step=True, on_epoch=True
        )
        self.log(
            "train_losses/pdg_loss", losses["loss_pdg"], on_step=True, on_epoch=True
        )

        return losses["loss"]

    def validation_step(self, batch, _batch_idx):
        outputs, targets, weights = self.forward(batch)
        target_kinematics, target_charge_cls, target_pdg_cls, target_mask = (
            self._extract_set_targets(targets)
        )

        losses = self.criterion(
            outputs=outputs,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_pdg_cls=target_pdg_cls,
            target_mask=target_mask,
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

        return {
            "pred_kinematics": outputs["pred_kinematics"],
            "pred_charge_logits": outputs["pred_charge_logits"],
            "pred_pdg_logits": outputs["pred_pdg_logits"],
            "pred_logits": outputs["pred_logits"],
            "pred_scores": object_scores,
            "pred_mask": pred_mask,
            "pred_charge": pred_charge,
            "pred_pdg": pred_pdg,
        }

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
