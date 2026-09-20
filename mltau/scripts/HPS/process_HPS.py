import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import multiprocessing
import random
import string

import awkward as ak
import hydra
from omegaconf import DictConfig

from mltau.models import HPS


@hydra.main(config_path="../../config", config_name="tau_builder", version_base=None)
def main(cfg: DictConfig) -> None:
    file_path = cfg.input_file_path
    data = ak.from_parquet(file_path)
    builder = HPS.HPSTauBuilder(cfg=cfg.models.HPS)
    processed_data = builder.process_jets(data)

    # Merge all input fields with all HPS-predicted tau properties
    data_to_save = {field: data[field] for field in data.fields}
    data_to_save.update(processed_data)

    file_name = os.path.basename(file_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    ak.to_parquet(ak.Array(data_to_save), os.path.join(cfg.output_dir, file_name))


if __name__ == "__main__":
    main()
