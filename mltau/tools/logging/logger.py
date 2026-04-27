import numpy as np
from omegaconf import DictConfig

from mltau.tools.logging import kinematics, decay_mode, charge_id, tagging


def log_all(
    targets: np.array,
    gen_jet_p4s: np.array,
    gen_jet_tau_p4s: np.array,
    reco_jet_p4s: np.array,
    predictions: np.array,
    cfg: DictConfig,
    tb_logger,
    current_epoch: int,
    dataset="train",
):
    tagging.log_all_tagging_metrics(
        targets=targets,
        gen_jet_p4s=gen_jet_p4s,
        gen_jet_tau_p4s=gen_jet_tau_p4s,
        reco_jet_p4s=reco_jet_p4s,
        predictions=predictions,
        cfg=cfg,
        tb_logger=tb_logger,
        current_epoch=current_epoch,
        dataset=dataset,
    )
    kinematics.log_all_kinematics_metrics(
        targets=targets,
        predictions=predictions,
        reco_jet_p4s=reco_jet_p4s,
        gen_jet_tau_p4s=gen_jet_tau_p4s,
        cfg=cfg,
        tb_logger=tb_logger,
        current_epoch=current_epoch,
        dataset=dataset,
    )
    decay_mode.log_all_decay_mode_metrics(
        targets=targets,
        predictions=predictions,
        tb_logger=tb_logger,
        current_epoch=current_epoch,
    )
    charge_id.log_charge_id_performance(
        targets=targets,
        predictions=predictions,
        gen_jet_tau_p4s=gen_jet_tau_p4s,
        reco_jet_p4s=reco_jet_p4s,
        cfg=cfg,
        tb_logger=tb_logger,
        current_epoch=current_epoch,
        dataset=dataset,
    )
