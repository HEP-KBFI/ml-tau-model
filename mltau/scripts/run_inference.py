import os
import hydra
import torch
from omegaconf import DictConfig
from mltau.models import MultiParTau_module, SingleParTau_module
from mltau.tools.evaluation import inference

@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """
    Standalone script to run inference on a model checkpoint.
    
    Usage:
        ./run.sh python3 mltau/scripts/run_inference.py \
            ckpt_path=/path/to/model.ckpt \
            output_dir=outputs/my_inference \
            [other config overrides]
    """
    # 1. Determine checkpoint path
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        # Default to the path used by train.py
        models_dir = os.path.join(cfg.output_dir, "models")
        ckpt_path = os.path.join(models_dir, "ParT-model_best.ckpt")
        print(f"[INFO] No ckpt_path provided, using default: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 2. Setup output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 3. Load model
    model_name = cfg.training.model.name
    print(f"[INFO] Loading {model_name} from {ckpt_path}")
    
    # Defaults consistent with train.py
    input_dim = 17
    num_dm_classes = 6
    
    if model_name == "MultiParTau":
        model = MultiParTau_module.ParTauModule.load_from_checkpoint(
            ckpt_path, cfg=cfg, input_dim=input_dim, num_dm_classes=num_dm_classes
        )
    elif model_name == "SingleParTau":
        model = SingleParTau_module.ParTauModule(
            cfg=cfg,
            input_dim=input_dim,
            num_dm_classes=num_dm_classes,
            task=cfg.training.model.task,
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        is_distilled = any(k.startswith("student.") for k in state_dict.keys())
        if is_distilled:
            print("[INFO] Distilled model detected. Mapping 'student.' to 'ParTau.' keys.")
            for k, v in state_dict.items():
                if k.startswith("student."):
                    new_state_dict[k.replace("student.", "ParTau.")] = v
        else:
            new_state_dict = state_dict
        model.load_state_dict(new_state_dict)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'MultiParTau' or 'SingleParTau'.")

    model.eval()
    
    # 4. Run inference
    # This generates .parquet files in {cfg.output_dir}/predictions/
    print(f"[INFO] Starting inference on test set...")
    inference.create_predictions_files(
        best_model=model, 
        model_name=model_name, 
        cfg=cfg,
        test_only=True
    )
    print(f"[INFO] Inference complete. Results saved to {os.path.join(cfg.output_dir, 'predictions')}")

if __name__ == "__main__":
    main()
