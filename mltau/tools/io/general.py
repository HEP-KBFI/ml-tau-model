import os
import glob
import json
import torch
import numpy as np
import awkward as ak
from dataclasses import dataclass
from torch.utils.data import Dataset


def get_all_paths(input_loc, n_files: int = None) -> list:
    """Loads all .parquet files specified by the input. The input can be a list of input_paths, a directory where the
    files are located or a wildcard path.

    Parameters:
        input_loc : str
            Location of the .parquet files.
        n_files : int
            [default: None] Maximum number of input files to be loaded. By default all will be loaded.
        columns : list
            [default: None] Names of the columns/branches to be loaded from the .parquet file. By default all columns
            will be loaded

    Returns:
        input_paths : list
            List of all the .parquet files found in the input location
    """
    if n_files == -1:
        n_files = None
    if isinstance(input_loc, list):
        input_paths = input_loc[:n_files]
    elif isinstance(input_loc, str):
        if os.path.isdir(input_loc):
            input_loc = os.path.expandvars(input_loc)
            input_paths = glob.glob(os.path.join(input_loc, "*.parquet"))[:n_files]
        elif "*" in input_loc:
            input_paths = glob.glob(input_loc)[:n_files]
        elif os.path.isfile(input_loc):
            input_paths = [input_loc]
        else:
            raise ValueError(f"Unexpected input_loc: {input_loc}")
    else:
        raise ValueError(f"Unexpected input_loc: {input_loc}")
    return input_paths


def get_row_groups(input_paths: list) -> list:
    """Get the row groups of the input files. The row groups are used to split the data into smaller chunks for
    processing.

    Parameters:
        input_paths : list
            List of all the .parquet files found in the input location

    Returns:
        row_groups : list
            List of all the row groups found in the input files
    """
    row_groups = []
    for data_path in input_paths:
        metadata = ak.metadata_from_parquet(data_path)
        num_row_groups = metadata["num_row_groups"]
        col_counts = metadata["col_counts"]
        row_groups.extend(
            [
                RowGroup(data_path, row_group, col_counts[row_group])
                for row_group in range(num_row_groups)
            ]
        )
    return row_groups


class RowGroupDataset(Dataset):
    def __init__(self, data_loc: str):
        self.data_loc = data_loc
        self.input_paths = get_all_paths(data_loc)
        self.row_groups = get_row_groups(self.input_paths)

    def __getitem__(self, index):
        return self.row_groups[index]

    def __len__(self):
        return len(self.row_groups)


class RowGroup:
    def __init__(self, filename, row_group, num_rows):
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows


def load_row_groups(filename):
    metadata = ak.metadata_from_parquet(filename)
    num_row_groups = metadata["num_row_groups"]
    col_counts = metadata["col_counts"]
    return [
        RowGroup(filename, row_group, col_counts[row_group])
        for row_group in range(num_row_groups)
    ]


def stack_and_pad_features(cand_features, max_cands):
    cand_features_tensors = np.stack(
        [
            ak.pad_none(cand_features[feat], max_cands, clip=True)
            for feat in cand_features.fields
        ],
        axis=-1,
    )
    cand_features_tensors = ak.to_numpy(ak.fill_none(cand_features_tensors, 0))
    # Swapping the axes such that it has the shape of (nJets, nFeatures, nParticles)
    cand_features_tensors = np.swapaxes(cand_features_tensors, 1, 2)

    cand_features_tensors[np.isnan(cand_features_tensors)] = 0
    cand_features_tensors[np.isinf(cand_features_tensors)] = 0
    return cand_features_tensors


class NpEncoder(json.JSONEncoder):
    """Class for encoding various objects such that they could be saved to a json file"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_to_json(data, output_path):
    """Saves data to a .json file located at `output_path`

    Args:
        data : dict
            The data to be saved
        output_path : str
            Destonation of the .json file

    Returns:
        None
    """
    with open(output_path, "wt") as out_file:
        json.dump(data, out_file, indent=4, cls=NpEncoder)


@dataclass
class BatchInputs:
    """Structured representation of batch inputs for better code readability."""

    cand_features: torch.Tensor
    cand_kinematics_pxpypze: torch.Tensor
    target: dict
    cand_mask: torch.Tensor
    weight: torch.Tensor
    gen_jet_tau_p4s: ak.Array
    reco_jet_p4s: ak.Array
    gen_jet_p4s: ak.Array


def load_all_data(
    input_loc: list, nfiles: int = None, columns: list = None, n_files: int = None
):
    input_files = get_all_paths(input_loc=input_loc, n_files=n_files)
    input_data = []
    # for file_path in tqdm.tqdm(sorted(input_files)):
    for i, file_path in enumerate(sorted(input_files)):
        print(f"[{i+1}/{len(input_files)}] Loading from {file_path}")
        try:
            input_data.append(load_parquet(file_path, columns=columns))
        except ValueError:
            print(f"{file_path} does not exist")
    if len(input_data) > 0:
        data = ak.concatenate(input_data)
        print("Input data loaded")
        return data
    else:
        raise ValueError("No data loaded")


def load_parquet(input_path: str, columns: list = None) -> ak.Array:
    """Loads the contents of the .parquet file specified by the input_path

    Args:
        input_path : str
            The path to the .parquet file to be loaded.
        columns : list
            Names of the columns/branches to be loaded from the .parquet file

    Returns:
        input_data : ak.Array
            The data from the .parquet file
    """
    ret = ak.from_parquet(input_path, columns=columns)
    ret = ak.Array({k: ret[k] for k in ret.fields})
    return ret
