import awkward as ak
import pytest
import torch
from omegaconf import OmegaConf

from mltau.tools.io.ParTauDETR_dataloader import (
    ParTauDETRDataModule,
    ParticleTransformerDETRDataset,
)
from mltau.tools.tau_id import TauIDClass, tau_id_classes
from mltau.tools.losses import TauLoss


def test_fixed_tau_id_class_indices():
    assert [(member.name.lower(), member.value) for member in TauIDClass] == [
        ("background", 0),
        ("hadronic_tau", 1),
        ("tau_electron", 2),
        ("tau_muon", 3),
    ]


@pytest.mark.parametrize(
    ("groups", "expected_names", "expected_labels"),
    [
        (
            {"background": ["qq", "mm", "bhabha"]},
            ("background", "hadronic_tau", "tau_electron", "tau_muon"),
            (0, 0, 0),
        ),
        (
            {"background": ["qq"], "lepton_background": ["mm", "bhabha"]},
            ("background", "hadronic_tau", "tau_electron", "tau_muon", "lepton_background"),
            (0, 4, 4),
        ),
        (
            {"background": ["qq"], "mm": ["mm"], "bhabha": ["bhabha"]},
            ("background", "hadronic_tau", "tau_electron", "tau_muon", "mm", "bhabha"),
            (0, 4, 5),
        ),
    ],
)
def test_background_labels(groups, expected_names, expected_labels):
    cfg = OmegaConf.create({"dataset": {"background_classes": groups}})
    names, samples = tau_id_classes(cfg)
    assert names == expected_names
    dataset = ParticleTransformerDETRDataset(row_groups=[], cfg=cfg)
    background = ak.Array([{"gen_jet_tau_decaymode": -1, "gen_jet_tau_vis_daughter_pdgs": []}])
    for sample, label in zip(("qq", "mm", "bhabha"), expected_labels):
        assert samples[sample] == label
        assert dataset._tau_id_targets(background, sample).tolist() == [label]

    signal = ak.Array([
        {"gen_jet_tau_decaymode": 0, "gen_jet_tau_vis_daughter_pdgs": [211]},
        {"gen_jet_tau_decaymode": 15, "gen_jet_tau_vis_daughter_pdgs": [11]},
        {"gen_jet_tau_decaymode": 15, "gen_jet_tau_vis_daughter_pdgs": [13]},
        {"gen_jet_tau_decaymode": -1, "gen_jet_tau_vis_daughter_pdgs": []},
    ])
    assert dataset._tau_id_targets(signal, "z").tolist() == [1, 2, 3, 0]
    with pytest.raises(ValueError, match="contains tau truth"):
        dataset._tau_id_targets(signal, "mm")


def test_discovery_only_selects_configured_samples(tmp_path):
    cfg = OmegaConf.create({
        "dataset": {"background_classes": {"background": ["qq", "mm"]}},
        "model": {"detr": {"tau_id_head": True}},
        "training": {"model": {"task": "set2set", "name": "ParTauDETR"}},
    })
    datamodule = ParTauDETRDataModule(cfg)
    for sample in ("z", "qq", "mm", "bhabha"):
        (tmp_path / f"{sample}_train_0000.parquet").touch()
    paths = datamodule._resolve_input_paths(str(tmp_path / "*_train*.parquet"), "train")
    assert {path.rsplit("/", 1)[-1] for path in paths} == {
        "z_train_0000.parquet", "qq_train_0000.parquet", "mm_train_0000.parquet"
    }
    (tmp_path / "mm_train_0000.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="mm"):
        datamodule._resolve_input_paths(str(tmp_path / "*_train*.parquet"), "train")


def test_invalid_background_configuration():
    for groups in (
        {"background": ["qq"], "mm": ["qq"]},
        {"background": ["z"]},
        {"background": []},
        {"mm": ["mm"]},
    ):
        with pytest.raises(ValueError):
            tau_id_classes(OmegaConf.create({"dataset": {"background_classes": groups}}))


def test_read_unit_passes_sample_to_detr_targets(monkeypatch):
    cfg = OmegaConf.create({"dataset": {"background_classes": {"background": ["qq"]}}})
    dataset = ParticleTransformerDETRDataset(row_groups=[], cfg=cfg)
    data = ak.Array([{"gen_jet_tau_decaymode": -1, "gen_jet_tau_vis_daughter_pdgs": []}])
    observed = []
    monkeypatch.setattr(dataset, "build_tensors", lambda rows, sample=None: observed.append(sample))
    dataset._build_read_tensors(data, "/tmp/qq_train_0000.parquet")
    assert observed == ["qq"]


def test_tau_id_loss_uses_per_jet_weights_for_six_classes():
    logits = torch.tensor([[3., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 2., 0.],
                           [0., 0., 0., 0., 0., 2.]])
    labels = torch.tensor([0, 4, 5])
    weights = torch.tensor([0.25, 1., 2.])
    expected = (torch.nn.functional.cross_entropy(logits, labels, reduction="none") * weights).sum() / weights.sum()
    assert torch.allclose(TauLoss(label_smoothing=0).compute_tagging_loss(logits, labels, weights), expected)