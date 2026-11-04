import vector
import numpy as np
import awkward as ak


def reinitialize_p4(p4_obj: ak.Array):
    """Reinitialized the 4-momentum for particle in order to access its properties.

    Args:
        p4_obj : ak.Array
            The particle represented by its 4-momenta

    Returns:
        p4 : ak.Array
            Particle with initialized 4-momenta, normalized to (rho, eta, phi, t).
    """
    # Normalise field names (aliases → canonical).
    name_map = {
        "rho": "pt",
        "x": "px",
        "y": "py",
        "z": "pz",
        "t": "energy",
        "e": "energy",
        "E": "energy",
        "tau": "mass",
        "m": "mass",
    }
    renamed = {name_map.get(f, f): p4_obj[f] for f in p4_obj.fields}

    # Pick the first complete non-redundant basis present in the data.
    # Mixing cylindrical (pt) and Cartesian (px/py) triggers vector's
    # "duplicate coordinates through momentum-aliases" error.
    for basis in (
        ("pt", "eta", "phi", "energy"),
        ("pt", "eta", "phi", "mass"),
        ("pt", "theta", "phi", "energy"),
        ("pt", "theta", "phi", "mass"),
        ("px", "py", "pz", "energy"),
        ("px", "py", "pz", "mass"),
    ):
        if all(k in renamed for k in basis):
            coords = {k: renamed[k] for k in basis}
            break
    else:
        raise ValueError(
            f"No supported 4-vector basis found in fields: {list(renamed)}"
        )

    p4 = vector.awk(ak.zip(coords))
    # Always return in (pt, eta, phi, energy) so downstream code can rely on
    # these being stored fields, not just computed properties.
    return vector.awk(
        ak.zip({"pt": p4.pt, "eta": p4.eta, "phi": p4.phi, "energy": p4.energy})
    )


def get_reduced_decaymodes(decaymodes: np.array):
    """Maps the full set of decay modes into a smaller subset, setting the rarer decaymodes under "Other" (# 15)"""
    target_mapping = {
        -1: 15,  # As we are running DM classification only on signal sample, then HPS_dm of -1 = 15 (Rare)
        0: 0,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 10,
        6: 11,
        7: 11,
        8: 11,
        9: 11,
        10: 10,
        11: 11,
        12: 11,
        13: 11,
        14: 11,
        15: 15,
        16: 16,
    }
    return np.vectorize(target_mapping.get)(decaymodes)


# Class order for the 6-class reduction of `gen_jet_tau_decaymode` that
# get_reduced_decaymodes produces; the default of prepare_one_hot_encoding /
# one_hot_decoding below.
STANDARD_DECAY_MODE_CLASSES = [0, 1, 2, 10, 11, 15]

# Class order for `gen_jet_tau_decaymode_rare` (see
# ntupelizer.tools.tau_decaymode.classify_rare_decay_mode): the fine-grained
# scheme that keeps kaon-bearing and other sub-modes separate instead of
# lumping them into "Rare" (as get_reduced_decaymodes does for the standard
# gen_jet_tau_decaymode). Order matters: it fixes the one-hot index of every
# class, so it must not change independently of a trained model's head.
RARE_DECAY_MODE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]


def get_decaymodes_rare_for_ohe(decaymodes_rare: np.array):
    """Maps -1 (no genuine tau, i.e. a background jet) onto 15 ("other"), so
    `gen_jet_tau_decaymode_rare` can be one-hot encoded without -1 crashing
    `prepare_one_hot_encoding`. The decay-mode loss is only ever computed on
    the is_tau mask, so what a background jet maps to here never reaches it.
    """
    return np.where(decaymodes_rare == -1, 15, decaymodes_rare)


# Looked up by `training.model.decay_mode_scheme` wherever a decay-mode class
# list has to be picked from config (e.g. mltau.tools.evaluation.inference).
DECAY_MODE_CLASS_SCHEMES = {
    "standard": STANDARD_DECAY_MODE_CLASSES,
    "rare": RARE_DECAY_MODE_CLASSES,
}


def prepare_one_hot_encoding(values, classes=STANDARD_DECAY_MODE_CLASSES):
    mapping = {class_: i for i, class_ in enumerate(classes)}
    return np.vectorize(mapping.get)(values)


def one_hot_decoding(values, classes=STANDARD_DECAY_MODE_CLASSES):
    mapping = {i: class_ for i, class_ in enumerate(classes)}
    return np.vectorize(mapping.get)(values)
