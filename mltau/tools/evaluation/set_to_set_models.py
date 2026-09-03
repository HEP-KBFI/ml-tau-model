import awkward as ak

from mltau.tools.general import reinitialize_p4

charged_pdg = [321, 211, 323]
neutral_pdg = [311, 221, 111, 130]


def count_ch_neutral(pdg):
    charged = ak.sum((abs(pdg) == 321) | (abs(pdg) == 211) | (abs(pdg) == 323), axis=1)

    neutral = ak.sum(
        (abs(pdg) == 311) | (abs(pdg) == 221) | (abs(pdg) == 111) | (abs(pdg) == 130),
        axis=1,
    )
    return charged, neutral


def get_decay_mode(n_charged, n_neutral):
    decay_mode = 5 * (n_charged - 1) + n_neutral
    return decay_mode


def construct_jet_level_predictions(pred_daughters, true_daughters):
    n_charged, n_neutral = count_ch_neutral(pred_daughters.pdg)
    pred_tau_decay_mode = get_decay_mode(n_charged, n_neutral)
    pred_tau_p4 = reinitialize_p4(ak.sum(pred_daughters.p4, axis=1))
    pred_tau_charge = ak.sum(pred_daughters.charge, axis=1)

    n_charged_true, n_neutral_true = count_ch_neutral(true_daughters.pdg)
    true_tau_decay_mode_exp = get_decay_mode(n_charged_true, n_neutral_true)
    return ak.Array(
        {
            "tau_decaymode": pred_tau_decay_mode,
            "tau_p4": pred_tau_p4,
            "tau_charge": pred_tau_charge,
            "gen_jet_tau_decaymode_exp": true_tau_decay_mode_exp,
        }
    )


def construct_prediction_file_content(data, pred_daughters, true_daughters):
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
            "pred_tau_daughter_pdgs": pred_daughters.pdg,
            "pred_tau_daughter_p4s": pred_daughters.p4,
            "pred_tau_daughter_charges": pred_daughters.charge,
        }
    )
    pred_tau_jet_level_data = construct_jet_level_predictions(
        pred_daughters, true_daughters
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
