import torch
import sys
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
                "batch_size": 128,
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
    assert batch[0].shape == (128, 17, 20)
    assert batch[1].shape == (128, 4, 20)
    assert isinstance(batch[2], dict)
    assert batch[3].shape == (128, 1, 20)
    assert batch[4].shape == (128,)
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
    assert batch[0].shape == (128, 17, 20)
    assert batch[1].shape == (128, 4, 20)
    assert isinstance(batch[2], dict)
    assert batch[3].shape == (128, 1, 20)
    assert batch[4].shape == (128,)
    print("OK")

def test_dataloader_consistency(cfg):
    print("Testing dataloader consistency...")
    dm_raw = ParTDataModuleRaw(cfg)
    dm_raw.setup("test")
    batch_raw = next(iter(dm_raw.test_dataloader()))
    
    dm_pre = ParTDataModulePre(cfg)
    dm_pre.setup("test")
    batch_pre = next(iter(dm_pre.test_dataloader()))
    
    assert len(batch_raw) == len(batch_pre)
    assert batch_raw[0].shape == batch_pre[0].shape
    assert batch_raw[1].shape == batch_pre[1].shape
    assert set(batch_raw[2].keys()) == set(batch_pre[2].keys())
    assert batch_raw[3].shape == batch_pre[3].shape
    assert batch_raw[4].shape == batch_pre[4].shape
    print("OK")

if __name__ == "__main__":
    cfg = get_cfg()
    try:
        test_raw_dataloader_structure(cfg)
        test_pre_dataloader_structure(cfg)
        test_dataloader_consistency(cfg)
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
