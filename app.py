import gradio as gr
import plotly.graph_objects as go
import numpy as np
import torch
import awkward as ak
import os
from omegaconf import OmegaConf
from mltau.models.MultiParTau_module import ParTauModule
from mltau.tools import general as g

# Load model and configs
base_path = os.path.join(os.path.dirname(__file__), "hf_space")

def load_model():
    checkpoint = os.path.join(base_path, "models/model.ckpt")
    import mltau
    mltau_root = os.path.dirname(mltau.__file__)
    config_root = os.path.join(mltau_root, "config")
    
    training_cfg = OmegaConf.load(os.path.join(config_root, "training.yaml"))
    dataset_cfg = OmegaConf.load(os.path.join(config_root, "dataset.yaml"))
    metrics_cfg = OmegaConf.load(os.path.join(config_root, "metrics/metrics.yaml"))
    metrics_cfg.tagging = OmegaConf.load(os.path.join(config_root, "metrics/tagging.yaml")).tagging
    metrics_cfg.kinematics = OmegaConf.load(os.path.join(config_root, "metrics/kinematics.yaml")).kinematics
    cfg = OmegaConf.merge(training_cfg, dataset_cfg, {"metrics": metrics_cfg})
    
    model = ParTauModule.load_from_checkpoint(
        checkpoint,
        cfg=cfg,
        input_dim=17,
        num_dm_classes=6,
        map_location=torch.device('cpu'),
        weights_only=False
    )
    model.eval()
    return model, cfg

model, cfg = load_model()

# --- Data Loading & Pre-computation ---
# The subset files were created by taking the first 10,000 jets from the 0509_dsinphi_to_sindphi dataset:
# ak.to_parquet(ak.from_parquet("0509_dsinphi_to_sindphi/z_test.parquet")[:10000], "hf_space/data/z_test_subset.parquet")
# ak.to_parquet(ak.from_parquet("0509_dsinphi_to_sindphi/qq_test.parquet")[:10000], "hf_space/data/qq_test_subset.parquet")
z_events = ak.from_parquet(os.path.join(base_path, "data/z_test_subset.parquet"))
qq_events = ak.from_parquet(os.path.join(base_path, "data/qq_test_subset.parquet"))

def run_full_inference(events):
    from mltau.tools.io.ParT_dataloader import ParticleTransformerDataset
    from mltau.tools.io.scaling import apply_saved_input_scaling_from_cfg
    
    # Ensure output_dir is defined for scaler path resolution if needed
    if "output_dir" not in cfg:
        cfg.output_dir = base_path
        
    ds = ParticleTransformerDataset([], cfg, batch_size=256)
    all_preds = {}
    all_features = []
    all_masks = []
    with torch.no_grad():
        for i in range(0, len(events), 256):
            chunk = events[i:i+256]
            tensors = ds.build_tensors(chunk)
            
            # Apply scaling (it will check if enabled in cfg)
            tensors = apply_saved_input_scaling_from_cfg(tensors, cfg)
            
            all_features.append(tensors[0].numpy())
            all_masks.append(tensors[3].numpy().squeeze(1))
            logits = model.ParTau(tensors[0].float(), tensors[1].float(), tensors[3].float())
            preds = model._convert_logits_to_predictions(logits)
            for k, v in preds.items():
                if k not in all_preds: all_preds[k] = []
                all_preds[k].append(v.cpu().numpy())
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k])
    return all_preds, np.concatenate(all_features), np.concatenate(all_masks)

print("Pre-computing inference results...")
z_results, z_feat, z_mask_feat = run_full_inference(z_events)
qq_results, qq_feat, qq_mask_feat = run_full_inference(qq_events)

FEATURE_NAMES = [
    "Δη (relative to jet)", "Δφ (relative to jet)", "log(pT)", "log(energy)", 
    "log(pT / jet pT)", "log(energy / jet energy)", "ΔR (relative to jet)", 
    "Charge", "isElectron", "isMuon", "isPhoton", "isChargedHadron", "isNeutralHadron", 
    "dz", "dz error", "dxy", "dxy error"
]

def get_properties(events):
    pdgs = events.reco_cand_pdgs
    return {
        "pt": ak.to_numpy(events.reco_jet_p4.rho),
        "n_total": ak.to_numpy(ak.num(pdgs)),
        "n_ch": ak.to_numpy(ak.sum((pdgs == 211) | (pdgs == -211), axis=1)),
        "n_nh": ak.to_numpy(ak.sum((pdgs == 130) | (pdgs == 2112), axis=1)),
        "n_photon": ak.to_numpy(ak.sum(pdgs == 22, axis=1)),
    }

z_props = get_properties(z_events)
qq_props = get_properties(qq_events)

# --- Dynamic Evaluation ---
from mltau.tools.evaluation.tagging import TaggerEvaluator
from mltau.tools.evaluation.kinematics import KinematicsEvaluator

def get_mask(props, min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph):
    return (
        (props["pt"] >= min_pt) & (props["pt"] <= max_pt) &
        (props["n_total"] >= min_tot) & (props["n_total"] <= max_tot) &
        (props["n_ch"] >= min_ch) & (props["n_ch"] <= max_ch) &
        (props["n_nh"] >= min_nh) & (props["n_nh"] <= max_nh) &
        (props["n_photon"] >= min_ph) & (props["n_photon"] <= max_ph)
    )

def update_scientific_plots(min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph):
    z_mask = get_mask(z_props, min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph)
    qq_mask = get_mask(qq_props, min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph)
    
    if not np.any(z_mask) or not np.any(qq_mask):
        empty = go.Figure().update_layout(title="No data matching filters", template="plotly_white")
        return empty, empty, empty, empty, empty, empty, empty
    
    # 1. ROC
    evaluator = TaggerEvaluator(
        signal_predictions=z_results["is_tau"][z_mask],
        signal_gen_tau_p4=z_events.gen_jet_tau_p4[z_mask],
        signal_reco_jet_p4=z_events.reco_jet_p4[z_mask],
        bkg_predictions=qq_results["is_tau"][qq_mask],
        bkg_gen_jet_p4=qq_events.gen_jet_p4[qq_mask],
        bkg_reco_jet_p4=qq_events.reco_jet_p4[qq_mask],
        cfg=cfg, sample="Filtered", algorithm="MultiParTau"
    )
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=evaluator.efficiencies, y=evaluator.fakerates, mode='lines', name='MultiParTau'))
    fig_roc.update_layout(title="Tau identification ROC curve", xaxis_title="Signal efficiency", yaxis_title="Background misidentification probability", yaxis_type="log", template="plotly_white", height=400)
    
    # 2. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    dm_mask = z_mask & (z_events.gen_jet_tau_decaymode != -1)
    if np.any(dm_mask):
        true_dm = g.get_reduced_decaymodes(ak.to_numpy(z_events.gen_jet_tau_decaymode[dm_mask]))
        true_dm_idx = g.prepare_one_hot_encoding(true_dm)
        pred_dm = np.argmax(z_results["decay_mode"][dm_mask], axis=1)
        cm = confusion_matrix(true_dm_idx, pred_dm, labels=range(6), normalize='true')
        
        # Use descriptive labels for the 6 classes
        class_names = ["1p 0pi0", "1p 1pi0", "1p >=2pi0", "3p 0pi0", "3p >=1pi0", "Rare"]
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names, colorscale='Viridis'))
        fig_cm.update_layout(
            title="Decay mode confusion matrix",
            template="plotly_white",
            height=400, width=400,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis_title="Predicted decay mode",
            yaxis_title="True decay mode"
        )
    else:
        fig_cm = go.Figure().update_layout(
            title="No signal decay modes in selection",
            template="plotly_white",
            height=400, width=400
        )

    # 3. Resolution
    def recon_p4(reg, reco_p4):
        pt_corr, eta_corr = np.exp(reg[:, 0]), reg[:, 1]
        phi_corr, m_corr = np.arctan2(reg[:, 2], reg[:, 3]), np.exp(reg[:, 4])
        pred_pt = reco_p4.rho * pt_corr
        pred_eta = reco_p4.eta + eta_corr
        pred_phi = reco_p4.phi + phi_corr
        reco_mass = np.sqrt(np.maximum(reco_p4.t**2 - (reco_p4.rho * np.cosh(reco_p4.eta))**2, 0.0))
        pred_mass = reco_mass * m_corr
        return ak.zip({"pt": pred_pt, "eta": pred_eta, "phi": pred_phi, "mass": pred_mass})

    sig_pred_p4 = recon_p4(z_results["kinematics"][z_mask], z_events.reco_jet_p4[z_mask])
    kin_eval = KinematicsEvaluator(predicted_p4=sig_pred_p4, true_p4=z_events.gen_jet_tau_p4[z_mask], cfg=cfg, algorithm="MultiParTau", sample_name="z")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=kin_eval.var_evaluators["pt"].bin_centers, y=kin_eval.var_evaluators["pt"].resolutions, mode='lines+markers', name='pT resolution'))
    fig_res.update_layout(title="Kinematic resolution", xaxis_title="gen pT [GeV]", yaxis_title="Resolution (IQR/Median)", template="plotly_white", height=400)

    # 4. Resolution Histograms
    def create_res_hist(data, title, xlabel, range=None):
        if range is not None:
            # Filter data to the specified range to ensure Plotly's nbinsx 
            # applies specifically to the visible area.
            data = data[(data >= range[0]) & (data <= range[1])]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=200, marker_color='#1f77b4', opacity=0.7))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Count", template="plotly_white", bargap=0.01, height=300)
        if range: fig.update_xaxes(range=range)
        return fig

    fig_res_pt = create_res_hist(kin_eval.var_evaluators["pt"].ratios, "pT Resolution Distribution", "pred pT / gen pT", [0.5, 1.5])
    fig_res_eta = create_res_hist(kin_eval.var_evaluators["eta"].ratios, "η Residual Distribution", "pred η - gen η", [-0.05, 0.05])
    fig_res_phi = create_res_hist(kin_eval.var_evaluators["phi"].ratios, "φ Residual Distribution", "pred φ - gen φ [deg]", [-1, 1])

    # 5. Discriminator Distribution
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Histogram(x=z_results["is_tau"][z_mask], name="Signal (Z)", marker_color="blue", opacity=0.6, nbinsx=50))
    fig_disc.add_trace(go.Histogram(x=qq_results["is_tau"][qq_mask], name="Background (QQ)", marker_color="red", opacity=0.6, nbinsx=50))
    fig_disc.update_layout(title="Tau identification score distribution", xaxis_title="Tau ID probability", yaxis_title="Count", barmode='overlay', template="plotly_white", height=400)

    return fig_roc, fig_cm, fig_res, fig_res_pt, fig_res_eta, fig_res_phi, fig_disc

def update_feature_distribution_plots(min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph):
    z_mask = get_mask(z_props, min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph)
    qq_mask = get_mask(qq_props, min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph)
    
    if not np.any(z_mask) or not np.any(qq_mask):
        return [go.Figure().update_layout(title="No data matching filters", template="plotly_white") for _ in FEATURE_NAMES]
    
    plots = []
    for i, name in enumerate(FEATURE_NAMES):
        z_vals = z_feat[z_mask, i, :][z_mask_feat[z_mask]]
        qq_vals = qq_feat[qq_mask, i, :][qq_mask_feat[qq_mask]]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=z_vals, name="Signal (Z)", marker_color="blue", opacity=0.6, nbinsx=100))
        fig.add_trace(go.Histogram(x=qq_vals, name="Background (QQ)", marker_color="red", opacity=0.6, nbinsx=100))
        fig.update_layout(
            title=f"{name}",
            xaxis_title=name,
            yaxis_title="Count",
            barmode='overlay',
            template="plotly_white",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        plots.append(fig)
    return plots

# --- Plotting Helpers ---
def plot_histogram(dataset_name, prop_key, min_val, max_val, title, xlabel):
    props = z_props if "Signal" in dataset_name else qq_props
    data = props[prop_key]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min_val, min_val, max_val, max_val], y=[0, 1, 1, 0], fill="toself", fillcolor="orange", opacity=0.1, line=dict(width=0), yaxis="y2"))
    fig.add_trace(go.Histogram(x=data, nbinsx=max(int(np.max(data)) + 1, 10) if prop_key != "pt" else 50, marker_color='#1f77b4', opacity=0.6))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Count", template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=300, showlegend=False, yaxis2=dict(overlaying='y', visible=False, range=[0, 1]))
    return fig

def filter_events_handler(*filters):
    z_mask = get_mask(z_props, *filters)
    qq_mask = get_mask(qq_props, *filters)
    z_indices = np.where(z_mask)[0].tolist()
    qq_indices = np.where(qq_mask)[0].tolist()
    
    status_text = f"**Event selection summary:**\n- Signal (Z): {len(z_indices)} / {len(z_events)}\n- Background (QQ): {len(qq_indices)} / {len(qq_events)}"
    
    z_update = gr.update(choices=z_indices, value=z_indices[0] if z_indices else None)
    qq_update = gr.update(choices=qq_indices, value=qq_indices[0] if qq_indices else None)
    return z_update, qq_update, status_text

def plot_2d_jet(idx, dataset_name):
    if idx is None: return go.Figure().update_layout(title="No event selected", template="plotly_white")
    events = z_events if "Signal" in dataset_name else qq_events
    event = events[idx]
    p4s, pdgs, jet_p4 = event.reco_cand_p4s, event.reco_cand_pdgs, event.reco_jet_p4
    d_eta, d_phi = p4s.eta - jet_p4.eta, np.arctan2(np.sin(p4s.phi - jet_p4.phi), np.cos(p4s.phi - jet_p4.phi))
    pdg_map = {11: "Electron", -11: "Electron", 13: "Muon", -13: "Muon", 22: "Photon", 211: "Charged Hadron", -211: "Charged Hadron", 130: "Neutral Hadron", 2112: "Neutral Hadron"}
    type_names = np.array([pdg_map.get(p, f"Other ({p})") for p in pdgs])
    unique_types = np.unique(type_names)
    color_palette = {"Charged Hadron": "#1f77b4", "Neutral Hadron": "#2ca02c", "Photon": "#ff7f0e", "Electron": "#17becf", "Muon": "#e377c2", "Other": "#7f7f7f"}
    fig = go.Figure()
    for t in unique_types:
        mask = type_names == t
        color = color_palette.get(t.split(" (")[0] if "Other" in t else t, color_palette["Other"])
        hover_text = [f"Type: {t}<br>pT: {pt:.2f} GeV<br>E: {en:.2f} GeV<br>dEta: {de:.3f}<br>dPhi: {dp:.3f}" for pt, en, de, dp in zip(p4s.rho[mask], p4s.t[mask], d_eta[mask], d_phi[mask])]
        fig.add_trace(go.Scatter(x=d_eta[mask], y=d_phi[mask], mode='markers', marker=dict(size=np.sqrt(p4s.t[mask])*4, color=color, line=dict(width=1, color='white')), text=hover_text, name=t, hoverinfo='text'))
    gt_de, gt_dp = 0, 0
    if event.gen_jet_tau_decaymode != -1:
        gt_p4 = event.gen_jet_tau_p4
        gt_de, gt_dp = gt_p4.eta - jet_p4.eta, np.arctan2(np.sin(gt_p4.phi - jet_p4.phi), np.cos(gt_p4.phi - jet_p4.phi))
        fig.add_trace(go.Scatter(x=[gt_de], y=[gt_dp], mode='markers', marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='white')), name="True Tau", hoverinfo='name'))

    # Calculate symmetric and square axis limits fitting all constituents
    limit = max(np.max(np.abs(d_eta)), np.max(np.abs(d_phi)), abs(gt_de), abs(gt_dp)) * 1.1
    limit = max(limit, 0.1)

    fig.update_layout(
        title=f"Jet Constituents ({dataset_name}, Event {idx})", 
        xaxis_title="Δη", yaxis_title="Δφ", 
        template="plotly_white", 
        xaxis=dict(range=[-limit, limit]),
        yaxis=dict(range=[-limit, limit], scaleanchor="x", scaleratio=1), 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600, width=600
    )
    return fig

def get_inference_display(idx, dataset_name):
    if idx is None: return "No event selected."
    results = z_results if "Signal" in dataset_name else qq_results
    events = z_events if "Signal" in dataset_name else qq_events
    event = events[idx]
    
    # Classification
    tau_prob = results["is_tau"][idx]
    charge = "+1" if results["charge"][idx] >= 0.5 else "-1"
    dm = [0, 1, 2, 10, 11, 15][np.argmax(results["decay_mode"][idx])]
    
    # Regression
    reg = results["kinematics"][idx]
    reco_p4 = event.reco_jet_p4
    pt_corr, eta_corr = np.exp(reg[0]), reg[1]
    phi_corr, m_corr = np.arctan2(reg[2], reg[3]), np.exp(reg[4])
    
    pred_pt = reco_p4.rho * pt_corr
    pred_eta = reco_p4.eta + eta_corr
    pred_phi = reco_p4.phi + phi_corr
    reco_mass = np.sqrt(np.maximum(reco_p4.t**2 - (reco_p4.rho * np.cosh(reco_p4.eta))**2, 0.0))
    pred_mass = reco_mass * m_corr

    # Ground Truth
    is_tau = "Yes" if event.gen_jet_tau_decaymode != -1 else "No"
    gt_c = f"{int(event.gen_jet_tau_charge):+d}" if is_tau == "Yes" else "N/A"
    gt_dm = f"{int(event.gen_jet_tau_decaymode)}" if is_tau == "Yes" else "N/A"
    
    if is_tau == "Yes":
        gt_p4 = event.gen_jet_tau_p4
        gt_mass = np.sqrt(np.maximum(gt_p4.t**2 - (gt_p4.rho * np.cosh(gt_p4.eta))**2, 0.0))
        gt_reg = f"- **pT**: {gt_p4.rho:.2f} GeV\n- **eta**: {gt_p4.eta:.3f}\n- **phi**: {gt_p4.phi:.3f}\n- **mass**: {gt_mass:.3f} GeV"
    else:
        gt_reg = "- **pT**: N/A\n- **eta**: N/A\n- **phi**: N/A\n- **mass**: N/A"

    return (
        f"### Prediction results\n"
        f"- **Tau ID score**: {tau_prob:.4f}\n"
        f"- **Predicted charge**: {charge}\n"
        f"- **Predicted decay mode**: {dm}\n"
        f"- **Predicted pT**: {pred_pt:.2f} GeV\n"
        f"- **Predicted eta**: {pred_eta:.3f}\n"
        f"- **Predicted phi**: {pred_phi:.3f}\n"
        f"- **Predicted mass**: {pred_mass:.3f} GeV\n\n"
        f"### Ground truth\n"
        f"- **Signal class**: {is_tau}\n"
        f"- **True charge**: {gt_c}\n"
        f"- **True decay mode**: {gt_dm}\n"
        f"{gt_reg}"
    )

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# FCC Tau Reconstruction with a Multi-Task Transformer")
    
    with gr.Tabs():
        with gr.TabItem("Dataset Selection"):
            dataset_name = gr.Radio(["Signal (Z)", "Background (QQ)"], value="Signal (Z)", label="Dataset (for Exploration View)")
            with gr.Row():
                with gr.Column():
                    pt_h = gr.Plot(value=plot_histogram("Signal (Z)", "pt", 0, 100, "Jet transverse momentum distribution", "pT [GeV]"))
                    min_pt, max_pt = gr.Slider(0, 100, 0, label="Minimum pT [GeV]"), gr.Slider(0, 100, 100, label="Maximum pT [GeV]")
                with gr.Column():
                    tot_h = gr.Plot(value=plot_histogram("Signal (Z)", "n_total", 0, 20, "Total Constituents", "N"))
                    min_tot, max_tot = gr.Slider(0, 20, 0, step=1, label="Minimum total constituents"), gr.Slider(0, 20, 20, step=1, label="Maximum total constituents")
                with gr.Column():
                    ch_h = gr.Plot(value=plot_histogram("Signal (Z)", "n_ch", 0, 10, "Charged Hadrons", "N"))
                    min_ch, max_ch = gr.Slider(0, 10, 0, step=1, label="Minimum charged hadrons"), gr.Slider(0, 10, 10, step=1, label="Maximum charged hadrons")
            with gr.Row():
                with gr.Column():
                    nh_h = gr.Plot(value=plot_histogram("Signal (Z)", "n_nh", 0, 10, "Neutral Hadrons", "N"))
                    min_nh, max_nh = gr.Slider(0, 10, 0, step=1, label="Minimum neutral hadrons"), gr.Slider(0, 10, 10, step=1, label="Maximum neutral hadrons")
                with gr.Column():
                    ph_h = gr.Plot(value=plot_histogram("Signal (Z)", "n_photon", 0, 15, "Photons", "N"))
                    min_ph, max_ph = gr.Slider(0, 15, 0, step=1, label="Minimum photons"), gr.Slider(0, 15, 15, step=1, label="Maximum photons")
                with gr.Column():
                    filter_status = gr.Markdown(value=f"Found {len(z_events)} events matching filters.")
            
            f_inputs = [min_pt, max_pt, min_tot, max_tot, min_ch, max_ch, min_nh, max_nh, min_ph, max_ph]

        with gr.TabItem("Performance Metrics"):
            with gr.Row(): roc_p, cm_p = gr.Plot(), gr.Plot()
            with gr.Row(): res_p, disc_p = gr.Plot(), gr.Plot()
            with gr.Row(): res_pt_p, res_eta_p, res_phi_p = gr.Plot(), gr.Plot(), gr.Plot()

        with gr.TabItem("Particle Feature Distributions"):
            gr.Markdown("Comparing distributions of the 17 input features for particles in signal vs. background jets.")
            feat_ps = []
            for i in range(0, 17, 3):
                with gr.Row():
                    for j in range(3):
                        if i + j < 17:
                            feat_ps.append(gr.Plot())

        with gr.TabItem("Event Visualization"):
            with gr.Column():
                gr.Markdown("### Signal Event (Z → ττ)")
                sig_event_idx = gr.Dropdown(choices=[i for i in range(len(z_events))], value=0, label="Select Signal Event Index")
                with gr.Row():
                    with gr.Column(scale=2): sig_jet_p = gr.Plot()
                    with gr.Column(scale=1): sig_pred_o = gr.Markdown()
            
            gr.HTML("<hr style='margin: 2em 0'>")
            
            with gr.Column():
                gr.Markdown("### Background Event (e+e- → qq)")
                bkg_event_idx = gr.Dropdown(choices=[i for i in range(len(qq_events))], value=0, label="Select Background Event Index")
                with gr.Row():
                    with gr.Column(scale=2): bkg_jet_p = gr.Plot()
                    with gr.Column(scale=1): bkg_pred_o = gr.Markdown()

    def sync_all(ds, *filters):
        sig_idx_up, bkg_idx_up, status = filter_events_handler(*filters)
        roc, cm, res, res_pt, res_eta, res_phi, disc = update_scientific_plots(*filters)
        hists = (
            plot_histogram(ds, "pt", filters[0], filters[1], "Jet pT Distribution", "pT [GeV]"),
            plot_histogram(ds, "n_total", filters[2], filters[3], "Total Constituents", "N"),
            plot_histogram(ds, "n_ch", filters[4], filters[5], "Charged Hadrons", "N"),
            plot_histogram(ds, "n_nh", filters[6], filters[7], "Neutral Hadrons", "N"),
            plot_histogram(ds, "n_photon", filters[8], filters[9], "Photons", "N")
        )
        feat_plots = update_feature_distribution_plots(*filters)
        return [sig_idx_up, bkg_idx_up, status, roc, cm, res, disc, res_pt, res_eta, res_phi] + list(hists) + feat_plots

    all_inputs = [dataset_name] + f_inputs
    all_outputs = [sig_event_idx, bkg_event_idx, filter_status, roc_p, cm_p, res_p, disc_p, res_pt_p, res_eta_p, res_phi_p, pt_h, tot_h, ch_h, nh_h, ph_h] + feat_ps
    
    demo.load(sync_all, inputs=all_inputs, outputs=all_outputs)
    for c in f_inputs: c.release(sync_all, inputs=all_inputs, outputs=all_outputs)
    dataset_name.change(sync_all, inputs=all_inputs, outputs=all_outputs)
    
    # Event Visualization Listeners
    sig_event_idx.change(lambda idx: plot_2d_jet(idx, "Signal (Z)"), inputs=[sig_event_idx], outputs=sig_jet_p)
    sig_event_idx.change(lambda idx: get_inference_display(idx, "Signal (Z)"), inputs=[sig_event_idx], outputs=sig_pred_o)
    bkg_event_idx.change(lambda idx: plot_2d_jet(idx, "Background (QQ)"), inputs=[bkg_event_idx], outputs=bkg_jet_p)
    bkg_event_idx.change(lambda idx: get_inference_display(idx, "Background (QQ)"), inputs=[bkg_event_idx], outputs=bkg_pred_o)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
