import torch
import sys
import tqdm
from omegaconf import OmegaConf
from mltau.tools.io.ParT_dataloader import ParTDataModule as ParTDataModuleRaw
from mltau.tools.io.preprocessed_ParTau_dataloader import ParTDataModule as ParTDataModulePre

def get_cfg():
    return OmegaConf.create({
        "dataset": {
            "data_dir": "0509_dsinphi_to_sindphi",
            "max_cands": 20,
            "relative_sizes": {"train": 0.7, "val": 0.1, "test": 0.2}
        },
        "training": {
            "model": {"task": "is_tau", "name": "MultiParTau"},
            "dataloader": {
                "batch_size": 1024,
                "num_dataloader_workers": 0,
                "prefetch_factor": 2
            },
            "input_scaling": {"enabled": False}
        }
    })

def test_raw_dataloader_structure(cfg):
    print("Testing Raw dataloader structure...")
    dm = ParTDataModuleRaw(cfg)
    dm.setup("test")
    loader = dm.test_dataloader()
    batch = next(iter(loader))
    
    assert len(batch) == 8
    # cand_features, cand_kinematics, targets, mask, weights, gen_tau, reco, gen_jet
    assert batch[0].shape[1:] == (17, 20)
    assert batch[1].shape[1:] == (4, 20)
    assert isinstance(batch[2], dict)
    assert batch[3].shape[1:] == (1, 20)
    assert len(batch[4].shape) == 1
    assert isinstance(batch[5], dict)
    assert isinstance(batch[6], dict)
    assert isinstance(batch[7], dict)
    print("OK")

def test_pre_dataloader_structure(cfg):
    print("Testing Preprocessed dataloader structure...")
    dm = ParTDataModulePre(cfg)
    dm.setup("test")
    loader = dm.test_dataloader()
    batch = next(iter(loader))
    
    assert len(batch) == 8
    assert batch[0].shape[1:] == (17, 20)
    assert batch[1].shape[1:] == (4, 20)
    assert isinstance(batch[2], dict)
    assert batch[3].shape[1:] == (1, 20)
    assert len(batch[4].shape) == 1
    print("OK")

def get_dataset_stats(loader, name):
    total_jets = 0
    total_signal = 0
    for batch in tqdm.tqdm(loader, desc=f"Iterating {name}"):
        is_tau = batch[2]["is_tau"]
        total_jets += is_tau.shape[0]
        total_signal += is_tau.sum().item()
    return total_jets, total_signal

def test_functional_equivalence(cfg):
    print("Testing functional equivalence...")
    
    dm_raw = ParTDataModuleRaw(cfg)
    dm_raw.setup("test")
    loader_raw = dm_raw.test_dataloader()
    
    dm_pre = ParTDataModulePre(cfg)
    dm_pre.setup("test")
    loader_pre = dm_pre.test_dataloader()
    
    n_raw, s_raw = get_dataset_stats(loader_raw, "Raw")
    n_pre, s_pre = get_dataset_stats(loader_pre, "Pre")
    
    print(f"Raw: {n_raw} jets, {s_raw} signal, fraction {s_raw/n_raw:.4f}")
    print(f"Pre: {n_pre} jets, {s_pre} signal, fraction {s_pre/n_pre:.4f}")
    
    assert n_raw == n_pre, f"Total jets mismatch: raw={n_raw}, pre={n_pre}"
    assert s_raw == s_pre, f"Signal count mismatch: raw={s_raw}, pre={s_pre}"
    
    # Verify no repetition/completeness if we know the expected number
    # From previous check: z_test (118787) + qq_test (786130) = 904917
    expected_total = 904917
    assert n_raw == expected_total, f"Expected {expected_total} jets, but got {n_raw}"
    
    print("OK")

if __name__ == "__main__":
    cfg = get_cfg()
    try:
        test_raw_dataloader_structure(cfg)
        test_pre_dataloader_structure(cfg)
        test_functional_equivalence(cfg)
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
