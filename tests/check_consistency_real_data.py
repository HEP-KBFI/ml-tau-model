import sys
import os
import torch
import numpy as np
import awkward as ak
from omegaconf import OmegaConf

# Add relevant paths to sys.path
# This script is in ml-tau-model/tests/
# We need to add ml-tau-model and ml-tau-data to sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
tau_root = os.path.abspath(os.path.join(test_dir, "..", ".."))

if os.path.join(tau_root, "ml-tau-model") not in sys.path:
    sys.path.append(os.path.join(tau_root, "ml-tau-model"))
if os.path.join(tau_root, "ml-tau-data") not in sys.path:
    sys.path.append(os.path.join(tau_root, "ml-tau-data"))

from mltau.tools.io.ParT_dataloader import ParticleTransformerDataset as LiveDataset
from mltau.tools.io.preprocessed_ParTau_dataloader import ParticleTransformerDataset as OfflineDataset
from ntupelizer.scripts.preprocess_torch import build_tensors_from_data
from mltau.tools.io import scaling

def get_real_data_path():
    path = os.path.join(tau_root, "ml-tau-model/0528_Large_stats/z_test.parquet")
    if not os.path.exists(path):
        return None
    return path

def test_real_data_consistency(real_data_path):
    # 1. Setup config
    cfg = OmegaConf.create({
        "dataset": {
            "max_cands": 20,
            "data_dir": os.path.dirname(real_data_path)
        },
        "training": {
            "dataloader": {
                "batch_size": 100
            },
            "input_scaling": {
                "enabled": False
            }
        }
    })
    max_cands = cfg.dataset.max_cands
    batch_size = cfg.training.dataloader.batch_size

    # 2. Load 100 jets
    data = ak.from_parquet(real_data_path)
    data = data[:100]

    # 3. Live path
    from mltau.tools.io.general import RowGroup
    mock_rg = RowGroup(filename=real_data_path, row_group=0, num_rows=100)
    live_ds = LiveDataset(row_groups=[mock_rg], cfg=cfg, batch_size=batch_size)
    
    live_tensors = live_ds.build_tensors(data)
    live_cand_features = live_tensors[0]

    # 4. Offline path (simulating preprocess_torch + load)
    raw_tensors = build_tensors_from_data(data, max_cands)
    offline_cand_features = raw_tensors[0].transpose(1, 2)
    
    offline_tensors = (
        offline_cand_features,
        raw_tensors[1].transpose(1, 2),
        raw_tensors[2],
        raw_tensors[3],
        raw_tensors[4],
        raw_tensors[5],
        raw_tensors[6],
        raw_tensors[7],
    )
    offline_ds = OfflineDataset(tensors=offline_tensors, batch_size=batch_size, shuffle=False)
    offline_batches = list(offline_ds)
    offline_batch = offline_batches[0]

    # 5. Comparisons
    assert live_cand_features.shape == offline_batch[0].shape
    torch.testing.assert_close(live_cand_features, offline_batch[0])

def test_real_data_consistency_with_scaling(real_data_path):
    scaler_path = os.path.join(test_dir, "dummy_scaler.npz")
    feature_indices = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16]
    
    # Create dummy scaler
    mean = np.zeros(len(feature_indices), dtype=np.float32)
    std = np.ones(len(feature_indices), dtype=np.float32)
    np.savez(scaler_path, mean=mean, std=std, feature_indices=np.array(feature_indices), feature_names=np.array(["dummy"]))

    cfg = OmegaConf.create({
        "dataset": {
            "max_cands": 20,
            "data_dir": os.path.dirname(real_data_path)
        },
        "training": {
            "dataloader": {
                "batch_size": 100
            },
            "input_scaling": {
                "enabled": True,
                "scaler_path": scaler_path,
                "continuous_feature_indices": feature_indices
            }
        }
    })
    batch_size = cfg.training.dataloader.batch_size

    try:
        data = ak.from_parquet(real_data_path)
        data = data[:100]

        from mltau.tools.io.general import RowGroup
        mock_rg = RowGroup(filename=real_data_path, row_group=0, num_rows=100)
        
        # Live path
        live_ds = LiveDataset(row_groups=[mock_rg], cfg=cfg, batch_size=batch_size)
        live_batches = list(live_ds)
        live_batch = live_batches[0]

        # Offline path
        raw_tensors = build_tensors_from_data(data, cfg.dataset.max_cands)
        offline_tensors_unscaled = (
            raw_tensors[0].transpose(1, 2),
            raw_tensors[1].transpose(1, 2),
            raw_tensors[2],
            raw_tensors[3],
            raw_tensors[4],
            raw_tensors[5],
            raw_tensors[6],
            raw_tensors[7],
        )
        offline_batch_scaled = scaling.apply_saved_input_scaling_from_cfg(offline_tensors_unscaled, cfg)

        # Compare
        torch.testing.assert_close(live_batch[0], offline_batch_scaled[0])

    finally:
        if os.path.exists(scaler_path):
            os.remove(scaler_path)

if __name__ == "__main__":
    # If run directly, just execute the tests
    p = get_real_data_path()
    if p:
        print(f"Running tests on {p}...")
        test_real_data_consistency(p)
        test_real_data_consistency_with_scaling(p)
        print("ALL TESTS PASSED!")
    else:
        print("Data not found, skipping.")
