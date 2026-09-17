import awkward as ak

from mltau.tools.general import reinitialize_p4
from mltau.tools.meson_classes import get_meson_classes

# Both lists hold hadrons only.  A daughter that is in neither -- a photon, a
# conversion electron, anything else -- is not on the prong/pi0 grid at all, so
# the jet is labelled rare rather than being silently counted as zero prongs.
charged_pdg = [321, 211, 323]
# 310 (K0_S) and 223 (omega) are neutral hadrons like the rest and must be here:
# leaving them out counted e.g. "K0_L pi K0_S" as one neutral instead of two, and
# "omega K" as zero instead of one, which shifted those decays a class down.
neutral_pdg = [311, 221, 111, 130, 310, 223]
hadron_pdg = charged_pdg + neutral_pdg

# Matches DM_NAME_MAPPING in ml-tau-data (ntupelizer/tools/tau_decaymode.py).
RARE_DECAY_MODE = 15


def _count_species(pdg, species):
    """Number of daughters per jet whose |PDG| is in `species`."""
    in_species = abs(pdg) == species[0]
    for code in species[1:]:
        in_species = in_species | (abs(pdg) == code)
    return ak.sum(in_species, axis=1)


def count_ch_neutral(pdg):
    """Charged- and neutral-hadron multiplicity per jet.

    Reads the species off the module-level lists rather than repeating them, so
    that adding a code in one place is enough.
    """
    return _count_species(pdg, charged_pdg), _count_species(pdg, neutral_pdg)


def count_non_hadrons(pdg):
    """Daughters that are neither a charged nor a neutral hadron.

    In this dataset that is photons from a radiative decay and the occasional
    conversion electron; neutrinos never appear, the daughter PDGs being the
    visible ones.
    """
    is_hadron = abs(pdg) == hadron_pdg[0]
    for code in hadron_pdg[1:]:
        is_hadron = is_hadron | (abs(pdg) == code)
    return ak.sum(~is_hadron, axis=1)


def get_decay_mode(n_charged, n_neutral, n_non_hadrons=None):
    """Map (charged, neutral) hadron multiplicity onto the decay mode id.

    A decay with a non-hadronic daughter is not on that grid, so when
    `n_non_hadrons` is given such jets are labelled rare instead of being
    assigned whatever class their hadrons alone would imply.
    """
    decay_mode = 5 * (n_charged - 1) + n_neutral
    if n_non_hadrons is not None:
        decay_mode = ak.where(n_non_hadrons > 0, RARE_DECAY_MODE, decay_mode)
    return decay_mode


def count_ch_neutral_meson_classes(meson_class, tau_daughter_pdg_ids):
    """Count charged and neutral daughters from configured class charges."""
    charged_indices = []
    neutral_indices = []
    for index, daughter_class in enumerate(get_meson_classes(tau_daughter_pdg_ids)):
        charges = set(daughter_class.charges)
        if charges == {0}:
            neutral_indices.append(index)
        elif 0 not in charges:
            charged_indices.append(index)
        else:
            raise ValueError(
                f"Meson class '{daughter_class.name}' mixes neutral and charged particles."
            )

    charged_mask = meson_class == charged_indices[0]
    for index in charged_indices[1:]:
        charged_mask = charged_mask | (meson_class == index)
    neutral_mask = meson_class == neutral_indices[0]
    for index in neutral_indices[1:]:
        neutral_mask = neutral_mask | (meson_class == index)
    return ak.sum(charged_mask, axis=1), ak.sum(neutral_mask, axis=1)


def construct_jet_level_predictions(
    pred_daughters, true_daughters, tau_daughter_pdg_ids
):
    n_charged, n_neutral = count_ch_neutral_meson_classes(
        pred_daughters.meson_class, tau_daughter_pdg_ids
    )
    pred_tau_decay_mode = get_decay_mode(n_charged, n_neutral)
    pred_tau_p4 = reinitialize_p4(ak.sum(pred_daughters.p4, axis=1))
    pred_tau_charge = ak.sum(pred_daughters.charge, axis=1)

    n_charged_true, n_neutral_true = count_ch_neutral_meson_classes(
        true_daughters.meson_class, tau_daughter_pdg_ids
    )
    true_tau_decay_mode_exp = get_decay_mode(n_charged_true, n_neutral_true)
    return ak.Array(
        {
            "tau_decaymode": pred_tau_decay_mode,
            "tau_p4": pred_tau_p4,
            "tau_charge": pred_tau_charge,
            "gen_jet_tau_decaymode_exp": true_tau_decay_mode_exp,
        }
    )


def construct_prediction_file_content(
    data, pred_daughters, true_daughters, tau_daughter_pdg_ids
):
    fields_of_interest = [
        "reco_jet_p4",
        "gen_jet_p4",
        "reco_cand_p4s",
        "reco_cand_pdgs",
        "reco_cand_charges",
        "gen_jet_tau_vis_energy",
        "gen_jet_tau_decaymode",
        "gen_jet_tau_charge",
        "gen_jet_tau_full_p4",
        "gen_jet_tau_vis_daughter_p4s",
        "gen_jet_tau_vis_daughter_pdgs",
        "gen_jet_tau_vis_daughter_charges",
        "gen_jet_tau_p4",
    ]
    data_of_interest = ak.Array(data[fields_of_interest])
    pred_tau_daughter_data = ak.Array(
        {
            "pred_tau_daughter_meson_classes": pred_daughters.meson_class,
            "pred_tau_daughter_p4s": pred_daughters.p4,
            "pred_tau_daughter_charges": pred_daughters.charge,
        }
    )
    pred_tau_jet_level_data = construct_jet_level_predictions(
        pred_daughters, true_daughters, tau_daughter_pdg_ids
    )
    combined_data = ak.zip(
        {
            **{f: data_of_interest[f] for f in ak.fields(data_of_interest)},
            **{f: pred_tau_daughter_data[f] for f in ak.fields(pred_tau_daughter_data)},
            **{
                f: pred_tau_jet_level_data[f]
                for f in ak.fields(pred_tau_jet_level_data)
            },
        },
        depth_limit=1,
    )
    return combined_data
