import torch
import numpy as np
import awkward as ak
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig
from mltau.tools import general as g

# from mltau.tools.optimizers.lookahead import Lookahead
from mltau.tools.io.general import BatchInputs

from mltau.tools.logging import logger
from mltau.models.MultiParTau import ParTau


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int):
        super().__init__()
        self.cfg = cfg
        self.ParTau = ParTau(
            input_dim=input_dim,
            num_dm_classes=num_dm_classes,  # Number of decay modes we wish to classify
            num_layers=2,  # cfg.models.ParticleTransformer.hyperparameters.num_layers,
            embed_dims=[
                256,
                512,
                256,
            ],  # cfg.models.ParticleTransformer.hyperparameters.embed_dims,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="theta-phi",
        )

        # Initialize loss functions once to avoid memory allocation overhead
        self.charge_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.tagging_loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.1
        )  # background=0, signal=1; label_smoothing reduces overconfidence oscillations
        self.decay_mode_loss = nn.CrossEntropyLoss(reduction="none")
        self.kinematics_loss = nn.HuberLoss(reduction="none", delta=1.0)

        self.num_tasks = 4
        # Disable automatic optimization so PCGrad can do per-task backward passes
        self.automatic_optimization = False
        self.phase2_initialized = False

    def training_step(self, batch, batch_idx):
        net_opt = self.optimizers()

        predictions, targets, sample_weights = self.forward(batch)

        # Per-task scalar losses: [tag, dm, charge, kin]
        task_losses = self._compute_per_task_losses(
            predictions, targets, sample_weights
        )

        # ------------------------------------------------------------------ #
        # PCGrad: for each task i, compute its gradient then subtract the     #
        # projection onto any task j whose gradient conflicts (dot < 0).      #
        # This eliminates destructive interference in the shared backbone      #
        # without requiring learned task weights.                              #
        #                                                                      #
        # AMP note: we scale each loss with the GradScaler before autograd.   #
        # grad so the scaler is properly initialised, then manually unscale   #
        # and step the raw optimizer — bypassing Lightning's wrapper which     #
        # would otherwise double-unscale and crash.                            #
        # ------------------------------------------------------------------ #
        # Only include parameters that are actually being trained — frozen params
        # (Phase 2) must be excluded so PCGrad does not write zero .grad values
        # into them, which would cause AdamW to treat them as having gradients.
        params = [p for p in self.ParTau.parameters() if p.requires_grad]

        # Apply per-task loss weights from config before PCGrad so that tasks
        # with higher weight have proportionally larger gradient magnitudes.
        # This makes it harder for PCGrad to cancel the weighted task's gradients
        # when conflicts arise (e.g. kinematics: 2.0 makes kin twice as resistant).
        # Note: unweighted losses are still stored in task_losses for logging.
        tw = self.cfg.training.task_weights
        task_weight_tensor = task_losses.new_tensor(
            [tw.tau_id, tw.decay_mode, tw.charge, tw.kinematics]
        )
        weighted_task_losses = task_losses * task_weight_tensor

        # 1. Collect per-task gradient vectors.
        # Use self.manual_backward() instead of torch.autograd.grad so that
        # Lightning's internal result collection is properly triggered on each
        # training step.  Without at least one manual_backward() call,
        # callback_metrics is never populated (automatic_optimization=False
        # skips the normal result-collection pathway), which silently prevents
        # ModelCheckpoint from saving.
        # manual_backward also handles AMP scaling internally, so no need to
        # call scaler.scale() manually here.
        task_grads = []
        for i in range(self.num_tasks):
            net_opt.zero_grad()
            self.manual_backward(
                weighted_task_losses[i], retain_graph=(i < self.num_tasks - 1)
            )
            grads = [
                (
                    p.grad.float().clone()
                    if p.grad is not None
                    else torch.zeros_like(p, dtype=torch.float32)
                )
                for p in params
            ]
            task_grads.append(grads)

        # 2. Project conflicting gradients (asymmetric: kinematics gradient is never
        # projected away from other tasks so its direction in the shared backbone is
        # always fully preserved.  Classification tasks are still projected away from
        # kinematics when they conflict, which further protects the kin gradient).
        KIN_IDX = self.num_tasks - 1  # kinematics is the last task
        pc_grads = [list(g) for g in task_grads]  # mutable copy
        for i in range(self.num_tasks):
            if i == KIN_IDX:  # never reduce the kinematics gradient
                continue
            for j in range(self.num_tasks):
                if i == j:
                    continue
                for k, (gi, gj) in enumerate(zip(pc_grads[i], task_grads[j])):
                    dot = (gi * gj).sum()
                    if dot < 0:  # conflict: remove component along gj
                        pc_grads[i][k] = gi - dot / (gj.norm().pow(2) + 1e-12) * gj

        # 3. Sum projected gradients and write into .grad
        for k, p in enumerate(params):
            merged = sum(pc_grads[i][k] for i in range(self.num_tasks))
            p.grad = merged.to(p.dtype)

        # 4. Clip → step.
        # manual_backward() already handled AMP scaling/unscaling internally.
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        net_opt.step()

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        combined_loss = task_losses.sum().detach()

        # Accumulate individual task losses for epoch-level logging
        with torch.no_grad():
            loss_dict = {
                "loss": combined_loss,
                "tau_id_loss": task_losses[0],
                "decay_mode_loss": task_losses[1],
                "charge_loss": task_losses[2],
                "kinematics_loss": task_losses[3],
            }
        for key, value in loss_dict.items():
            self.training_loss_accumulator[key].append(value.detach())

        self.log(
            "LR",
            net_opt.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return combined_loss

    def predict_step(self, batch, _batch_idx):
        """
        Runs inference for a batch during prediction (trainer.predict()).
        Returns only the predictions dict (no losses/metrics).
        """
        # Unpack batch if needed (BatchInputs or tuple)
        if isinstance(batch, (list, tuple)):
            # Standard tuple from DataLoader
            logits, _, _ = self.forward(batch)
        else:
            # Already a BatchInputs or similar
            logits, _, _ = self.forward((batch,))
        predictions = self._convert_logits_to_predictions(logits)
        return predictions

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def reinitialize_optimizer_for_phase2(self):
        """Freeze the shared backbone; fine-tune kinematics params + DM head.

        The DM head remains trainable in Phase 2 to allow it to continue 
        adapting as the shared backbone weights might be updated by kinematics,
        though they now use separate CLS tokens and blocks.

        Both steps are required:
          1. requires_grad_(False) on frozen params so PCGrad's `params` list
             excludes them (no spurious zero gradients written into them).
          2. Rebuild the AdamW optimizer with only the unfrozen params so that
             momentum buffers are not accumulated for frozen weights.
        """
        if self.phase2_initialized:
            return

        # 1. Freeze everything, then selectively unfreeze.
        self.ParTau.requires_grad_(False)
        kin_params = (
            list(self.ParTau.regression_head.parameters())
            + list(self.ParTau.cls_blocks_kinematics.parameters())
            + list(self.ParTau.norm_kinematics.parameters())
            + [self.ParTau.cls_token_kinematics]
        )
        # DM head must adapt as kinematics representation changes.
        dm_params = list(self.ParTau.classification_head.parameters())
        for p in kin_params + dm_params:
            p.requires_grad_(True)

        base_lr = self.cfg.training.lr
        kin_lr = base_lr * self.cfg.training.kinematics_lr_multiplier
        new_optimizer = torch.optim.AdamW(
            [
                {"params": kin_params, "lr": kin_lr},
                {"params": dm_params, "lr": base_lr},
            ],
            weight_decay=1e-2,
        )
        # Use a simple CosineAnnealingLR for the fine-tuning phase.
        # OneCycleLR is problematic when restarted mid-run as it expects to start from 0.
        remaining = max(
            1, self.trainer.estimated_stepping_batches - self.trainer.global_step
        )
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            new_optimizer,
            T_max=remaining,
            eta_min=1e-7,
        )
        self.trainer.optimizers = [new_optimizer]
        self.trainer.lr_scheduler_configs[0].scheduler = new_scheduler
        n_kin = sum(p.numel() for p in kin_params)
        n_dm = sum(p.numel() for p in dm_params)
        print(
            f"[Phase 2 @ epoch {self.current_epoch}] Backbone frozen. "
            f"Trainable: {n_kin:,} kin params + {n_dm:,} DM head params"
        )
        self.phase2_initialized = True

    def configure_optimizers(self):
        # AdamW is generally preferred for transformer architectures.
        # Split into two param groups: the kinematics-exclusive parameters
        # (regression_head + cls_token_kinematics) get a higher LR to help them
        # catch up with the other heads.  The backbone LR is unchanged — the
        # backbone gradient is the PCGrad-merged sum and is unaffected by
        # per-head learning rates.
        base_lr = self.cfg.training.lr
        kin_lr = base_lr * self.cfg.training.kinematics_lr_multiplier

        kin_param_ids = {
            id(p)
            for p in list(self.ParTau.regression_head.parameters())
            + list(self.ParTau.cls_blocks_kinematics.parameters())
            + list(self.ParTau.norm_kinematics.parameters())
            + [self.ParTau.cls_token_kinematics]
        }
        kin_params = [p for p in self.ParTau.parameters() if id(p) in kin_param_ids]
        other_params = [
            p for p in self.ParTau.parameters() if id(p) not in kin_param_ids
        ]

        net_optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": base_lr},
                {"params": kin_params, "lr": kin_lr},
            ]
        )
        # if self.cfg.training.optimizer.use_lookahead:
        #     optimizer = Lookahead(base_optimizer=optimizer, k=6, alpha=0.5)

        # Use a more reliable method to calculate T_max
        # Check if estimated_stepping_batches is available and valid
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if estimated_steps is None or estimated_steps <= 0:
            # Fallback: calculate based on config (will be approximate but functional)
            max_epochs = self.cfg.training.trainer.max_epochs
            # Use a conservative estimate of steps per epoch
            # This will be less precise but the scheduler will still work
            estimated_steps_per_epoch = 500  # Reasonable default for most datasets
            T_max = max_epochs * estimated_steps_per_epoch
            print(
                f"Warning: Using estimated T_max={T_max} (estimated_stepping_batches not available)"
            )
        else:
            T_max = estimated_steps
            print(f"Using calculated T_max={T_max} from estimated_stepping_batches")

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            net_optimizer,
            max_lr=[base_lr, kin_lr],
            total_steps=T_max,
            anneal_strategy="cos",
        )
        return [net_optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def _convert_logits_to_predictions(self, logits_dict):
        """Convert model logits to probabilities/predictions for evaluation and logging."""
        predictions = {}

        # Convert awkward arrays to tensors if needed, apply activations, then convert back
        for key, logits in logits_dict.items():
            # Convert awkward array to tensor if necessary
            if hasattr(logits, "to_numpy"):  # awkward array
                logits_tensor = torch.from_numpy(ak.to_numpy(logits))
            else:  # already a tensor
                logits_tensor = logits

            # Apply appropriate activation
            if key == "is_tau":  # Two-class head; return signal probability (class 1)
                pred_tensor = torch.softmax(logits_tensor, dim=-1)[:, 1]
            elif key == "charge":  # Binary classification head
                pred_tensor = torch.sigmoid(logits_tensor)
            elif key == "decay_mode":  # Multiclass classification
                pred_tensor = torch.softmax(logits_tensor, dim=-1)
            else:  # Regression (kinematics)
                pred_tensor = logits_tensor

            # Convert back to awkward array to match the expected format
            if hasattr(logits, "to_numpy"):  # Input was awkward array
                predictions[key] = ak.from_numpy(pred_tensor.detach().cpu().numpy())
            else:  # Input was tensor
                predictions[key] = pred_tensor

        return predictions

    def forward(self, batch):
        """Both `predictions` and `targets` are defined for the multiple heads"""
        # Unpack batch components
        inputs = BatchInputs(*batch)

        predictions = self.ParTau(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        return predictions, inputs.target, inputs.weight

    def charge_loss_fn(self, predictions, targets):
        return self.charge_loss(predictions, targets)

    def tagging_loss_fn(self, predictions, targets):
        return self.tagging_loss(predictions, targets.long())

    def decay_mode_loss_fn(self, predictions, targets):
        return self.decay_mode_loss(predictions, targets)

    def kinematics_loss_fn(self, predictions, targets, l_m: float = 0.2):
        # Log-ratio terms: independent Huber in log space
        log_pt_loss = self.kinematics_loss(predictions[:, 0], targets[:, 0])  # log pT
        log_m_loss = self.kinematics_loss(predictions[:, 4], targets[:, 4])  # log mass

        # Delta eta term
        deta_loss = self.kinematics_loss(predictions[:, 1], targets[:, 1])  # delta eta

        # Phi chord loss: treat (sin, cos) as a 2D unit-vector difference so that
        # the gradient corrects the angular direction jointly rather than pushing
        # sin and cos independently.  The chord distance equals 2|sin(Δφ/2)|.
        phi_chord_loss = torch.sqrt(
            (predictions[:, 2] - targets[:, 2]) ** 2
            + (predictions[:, 3] - targets[:, 3]) ** 2
            + 1e-8
        )  # shape [N], differentiable everywhere

        # 3 independent components + mass, normalise by sum of weights
        return (log_pt_loss + deta_loss + phi_chord_loss + l_m * log_m_loss) / (
            3.0 + l_m
        )

    def _compute_per_task_losses(self, predictions, targets, sample_weights):
        """
        Return a [4] tensor [tag_loss, dm_loss, charge_loss, kin_loss] — each a
        sample-weighted scalar with a gradient graph attached (for GradNorm).

        The three signal-only tasks (dm, charge, kin) require at least one tau jet
        in the batch; if there are none they are returned as zero tensors so that
        GradNorm can still run without crashing (their gradient norm will be 0 and
        their target G_target will be pulled toward 0 as well).
        """
        is_tau_mask = targets["is_tau"].bool()

        # Tagging loss — all jets
        tag_per_jet = self.tagging_loss_fn(predictions["is_tau"], targets["is_tau"])
        tag_loss = (tag_per_jet * sample_weights).mean()

        if not is_tau_mask.any():
            zero = tag_loss.new_zeros(())
            return torch.stack([tag_loss, zero, zero, zero])

        tau_weights = sample_weights[is_tau_mask]

        dm_loss = (
            self.decay_mode_loss_fn(
                predictions["decay_mode"][is_tau_mask],
                targets["decay_mode"][is_tau_mask],
            )
            * tau_weights
        ).mean()

        charge_loss = (
            self.charge_loss_fn(
                predictions["charge"][is_tau_mask],
                targets["charge"][is_tau_mask],
            )
            * tau_weights
        ).mean()

        kin_loss = (
            self.kinematics_loss_fn(
                predictions["kinematics"][is_tau_mask],
                targets["kinematics"][is_tau_mask],
            )
            * tau_weights
        ).mean()

        return torch.stack([tag_loss, dm_loss, charge_loss, kin_loss])

    def calculate_metrics(
        self, targets, predictions, weights, w_kin=1, w_dm=1, w_tag=1, w_charge=1
    ):
        is_tau_mask = targets["is_tau"].bool()

        # Per-jet losses — shape [N]
        tau_id_loss_per_jet = self.tagging_loss_fn(
            predictions["is_tau"], targets["is_tau"]
        )

        # Start combined per-jet loss with tagging term
        combined_per_jet = w_tag * tau_id_loss_per_jet

        if not is_tau_mask.any():
            return {
                "tau_id_loss": tau_id_loss_per_jet.mean(),
                "charge_loss": combined_per_jet.new_zeros(()),
                "decay_mode_loss": combined_per_jet.new_zeros(()),
                "kinematics_loss": combined_per_jet.new_zeros(()),
                "loss": (combined_per_jet * weights).mean(),
            }

        # Per-jet losses for signal-only heads — shape [N_signal]
        dm_loss_per_jet = self.decay_mode_loss_fn(
            predictions["decay_mode"][is_tau_mask], targets["decay_mode"][is_tau_mask]
        )
        charge_loss_per_jet = self.charge_loss_fn(
            predictions["charge"][is_tau_mask], targets["charge"][is_tau_mask]
        )
        kin_loss_per_jet = self.kinematics_loss_fn(
            predictions["kinematics"][is_tau_mask], targets["kinematics"][is_tau_mask]
        )

        # loss = torch.stack(
        #     [
        #         tau_id_loss_per_jet,
        #         dm_loss_per_jet,
        #         charge_loss_per_jet,
        #         kin_loss_per_jet,
        #     ]
        # )

        # Add signal-only terms into combined per-jet loss
        combined_per_jet[is_tau_mask] += (
            w_dm * dm_loss_per_jet
            + w_charge * charge_loss_per_jet
            + w_kin * kin_loss_per_jet
        )

        # Multiply each jet's combined loss by its cls_weight, then average
        loss = (combined_per_jet * weights).mean()

        return {
            "tau_id_loss": tau_id_loss_per_jet.mean(),
            "charge_loss": charge_loss_per_jet.mean(),
            "decay_mode_loss": dm_loss_per_jet.mean(),
            "kinematics_loss": kin_loss_per_jet.mean(),
            "loss": loss,
        }

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)

        inputs = BatchInputs(*batch)

        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )

        output = {
            "predictions": predictions,
            "targets": targets,
            # "weights": weights,
            "inputs": inputs,  # Store inputs for p4s extraction at epoch end
        }
        self.validation_outputs.append(output)
        for key, value in metrics.items():
            self.validation_loss_accumulator[key].append(value.detach())

        return metrics["loss"]

    def on_validation_epoch_start(self):
        """Initialize storage for validation outputs."""
        self.validation_outputs = []
        self.validation_loss_accumulator = {
            key: []
            for key in [
                "loss",
                "tau_id_loss",
                "charge_loss",
                "decay_mode_loss",
                "kinematics_loss",
                "tau_id_loss_weighted",
                "charge_loss_weighted",
                "decay_mode_loss_weighted",
                "kinematics_loss_weighted",
            ]
        }

    def _log_at_epoch_end(self, dataset: str):
        if dataset == "val" and self.trainer.sanity_checking:
            return

        dataset_outputs = (
            self.validation_outputs if dataset == "val" else self.training_outputs
        )

        if dataset_outputs:
            # Aggregate all predictions, targets, and weights
            all_predictions = {}
            all_targets = {}
            # all_weights = []
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}
            all_inputs = []  # Store all inputs for baseline calculation

            for output in dataset_outputs:
                # Concatenate predictions for each head
                for key, pred in output["predictions"].items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.detach().cpu())

                # Concatenate targets
                for key, target in output["targets"].items():
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(target.detach().cpu())

                # Store inputs for baseline calculation and p4s extraction
                inputs = output["inputs"]
                all_inputs.append(inputs)

                # Extract p4s from inputs
                for key, value in inputs.gen_jet_p4s.items():
                    if key not in all_gen_jet_p4s:
                        all_gen_jet_p4s[key] = []
                    all_gen_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in inputs.reco_jet_p4s.items():
                    if key not in all_reco_jet_p4s:
                        all_reco_jet_p4s[key] = []
                    all_reco_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in inputs.gen_jet_tau_p4s.items():
                    if key not in all_gen_jet_tau_p4s:
                        all_gen_jet_tau_p4s[key] = []
                    all_gen_jet_tau_p4s[key].append(ak.Array(value.detach().cpu()))

                # Concatenate weights
                # all_weights.append(output["weights"].detach().cpu())

            # Convert lists to tensors
            for key in all_predictions:
                all_predictions[key] = ak.concatenate(all_predictions[key], axis=0)
            for key in all_targets:
                all_targets[key] = ak.concatenate(all_targets[key], axis=0)

            # Convert logits to probabilities for logging (evaluation functions expect probabilities)
            all_predictions_for_logging = self._convert_logits_to_predictions(
                all_predictions
            )
            for key in all_gen_jet_p4s:
                all_gen_jet_p4s[key] = ak.concatenate(all_gen_jet_p4s[key], axis=0)
            for key in all_reco_jet_p4s:
                all_reco_jet_p4s[key] = ak.concatenate(all_reco_jet_p4s[key], axis=0)
            for key in all_gen_jet_tau_p4s:
                all_gen_jet_tau_p4s[key] = ak.concatenate(
                    all_gen_jet_tau_p4s[key], axis=0
                )

            # Convert dictionaries back to awkward arrays with fields for reinitialize_p4
            gen_jet_p4s = ak.Array(all_gen_jet_p4s)
            reco_jet_p4s = ak.Array(all_reco_jet_p4s)
            gen_jet_tau_p4s = ak.Array(all_gen_jet_tau_p4s)

            # Log comprehensive metrics with full validation dataset
            current_epoch = self.current_epoch
            tb_logger = self.logger.experiment
            logger.log_all(
                targets=all_targets,
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                predictions=all_predictions_for_logging,  # Use probabilities for logging
                cfg=self.cfg,
                tb_logger=tb_logger,
                current_epoch=current_epoch,
                dataset=dataset,
            )
            # Clear outputs to free memory
            dataset_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            epoch_metrics = {
                k: torch.stack(v).mean()
                for k, v in self.validation_loss_accumulator.items()
                if v
            }
            for k, v in epoch_metrics.items():
                self.log(f"val_losses/{k}", v)
        self._log_at_epoch_end(dataset="val")

    def on_train_epoch_start(self):
        # Automatically trigger phase-2 kinematics fine-tuning when configured.
        phase2_epoch = getattr(self.cfg.training, "phase2_start_epoch", None)
        if phase2_epoch is not None and self.current_epoch == phase2_epoch:
            if not self.phase2_initialized:
                self.reinitialize_optimizer_for_phase2()

        self.training_loss_accumulator = {
            key: []
            for key in [
                "loss",
                "tau_id_loss",
                "charge_loss",
                "decay_mode_loss",
                "kinematics_loss",
                "tau_id_loss_weighted",
                "charge_loss_weighted",
                "decay_mode_loss_weighted",
                "kinematics_loss_weighted",
            ]
        }

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
