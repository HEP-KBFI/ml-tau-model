import awkward as ak
import numpy as np
import torch

from mltau.tools import general as g
from mltau.tools.io.ParT_dataloader import ParTDataModule, ParticleTransformerDataset


class ParticleTransformerRareDecayDataset(ParticleTransformerDataset):
    """
    ParticleTransformerDataset with the `decay_mode` target replaced by the
    fine-grained `gen_jet_tau_decaymode_rare` scheme (13 classes: the 12
    explicit prong/pi0/kaon combinations plus "other"; see
    ntupelizer.tools.tau_decaymode.classify_rare_decay_mode) instead of the
    6-class reduction of `gen_jet_tau_decaymode` the base class uses.

    Every other input and target (kinematics, charge, is_tau) is unchanged.
    """

    _NEEDED_COLUMNS = ParticleTransformerDataset._NEEDED_COLUMNS + [
        "gen_jet_tau_decaymode_rare",
    ]

    NUM_DM_CLASSES = len(g.RARE_DECAY_MODE_CLASSES)

    def build_tensors(self, data: ak.Array):
        (
            cand_features,
            cand_kinematics,
            targets,
            mask,
            weights,
            gen_jet_tau_p4s,
            reco_jet_p4s,
            gen_jet_p4s,
        ) = super().build_tensors(data)

        rare_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode_rare)
        rare_decaymode_for_ohe = g.get_decaymodes_rare_for_ohe(rare_decaymode)
        dm_class_index = torch.from_numpy(
            g.prepare_one_hot_encoding(
                rare_decaymode_for_ohe, classes=g.RARE_DECAY_MODE_CLASSES
            ).astype(np.int64)
        )
        targets["decay_mode"] = torch.nn.functional.one_hot(
            dm_class_index, self.NUM_DM_CLASSES
        ).float()

        return (
            cand_features,
            cand_kinematics,
            targets,
            mask,
            weights,
            gen_jet_tau_p4s,
            reco_jet_p4s,
            gen_jet_p4s,
        )


class RareDecaysDataModule(ParTDataModule):
    """DataModule variant using ParticleTransformerRareDecayDataset."""

    dataset_cls = ParticleTransformerRareDecayDataset
