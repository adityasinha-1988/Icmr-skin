import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# --- 1. System Setup & Model Loading ---
st.set_page_config(page_title="Nano-Sunscreen Optimizer", layout="wide")

@st.cache_resource
def load_surrogate_models():
    try:
        return joblib.load('macroscopic_surrogate_models.pkl')
    except FileNotFoundError:
        return None

models = load_surrogate_models()

st.title("AI-Guided Nanoemulsion Formulation Portal")

if not models:
    st.error("Error: 'macroscopic_surrogate_models.pkl' nahi mili. Please ensure the model file is in your GitHub repository.")
    st.stop()

# Base Features and Targets
features = [
    'hydroxytyrosol_wt_pct', 'oleuropein_wt_pct', 
    'isopropyl_myristate_wt_pct', 'tween80_wt_pct', 
    'span80_wt_pct', 'aqueous_phase_wt_pct'
]
targets = ['spf', 'droplet_size_nm', 'viscosity_cp']

# --- 2. Bio-Team Data Integration (Sidebar Top) ---
st.sidebar.header("🧪 Lab Data Integration")

# 2A. Template Download (Updated with Metadata)
metadata_cols = ['experiment_id', 'date', 'time']
template_df = pd.DataFrame(columns=metadata_cols + features + targets)
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Blank CSV Template", 
    data=csv_template, 
    file_name="lab_data_template.csv", 
    mime="text/csv"
)

# 2B. Data Upload & Retraining
uploaded_file = st.sidebar.file_uploader("Upload Wet-Lab Results (.csv)", type=["csv"])
if uploaded_file is not None:
    if st.sidebar.button("Retrain AI Models Now"):
        with st.spinner("Retraining XGBoost on real lab data..."):
            try:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from xgboost import XGBRegressor
                
                lab_df = pd.read_csv(uploaded_file)
                
                # Validation check
                if not all(col in lab_df.columns for col in features + targets):
                    st.sidebar.error("CSV columns mismatch. Use the downloaded template exactly.")
                else:
                    X_lab = lab_df[features]
                    new_models = {}
                    
                    for target in targets:
                        y_lab = lab_df[target]
                        pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('regressor', XGBRegressor(random_state=42, n_estimators=200, max_depth=4, learning_rate=0.05))
                        ])
                        pipe.fit(X_lab, y_lab)
                        new_models[target] = pipe
                    
                    joblib.dump(new_models, 'macroscopic_surrogate_models_UPDATED.pkl')
                    st.sidebar.success("Models retrained successfully!")
                    
                    with open('macroscopic_surrogate_models_UPDATED.pkl', 'rb') as f:
                        st.sidebar.download_button(
                            label="Download Updated Model (.pkl)",
                            data=f,
                            file_name="macroscopic_surrogate_models.pkl",
                            mime="application/octet-stream"
                        )
                    st.sidebar.info("Bio-team: Send this downloaded .pkl file to the CS team to permanently update the portal.")
            except Exception as e:
                st.sidebar.error(f"Retraining failed: {e}")

st.sidebar.markdown("---")

# --- 3. Interactive Inputs & Normalization ---
st.sidebar.header("Formulation Inputs (Raw wt%)")
ht = st.sidebar.slider("Hydroxytyrosol", 0.1, 5.0, 1.0, 0.1)
ol = st.sidebar.slider("Oleuropein", 0.1, 5.0, 2.0, 0.1)
ipm = st.sidebar.slider("Isopropyl Myristate (Oil)", 10.0, 30.0, 20.0, 0.5)
t80 = st.sidebar.slider("Tween 80", 2.0, 10.0, 5.0, 0.1)
s80 = st.sidebar.slider("Span 80", 1.0, 5.0, 2.5, 0.1)
aq = st.sidebar.slider("Aqueous Phase", 40.0, 90.0, 69.5, 0.5)

raw_inputs = np.array([ht, ol, ipm, t80, s80, aq])
normalized_inputs = (raw_inputs / raw_inputs.sum()) * 100
input_df = pd.DataFrame([normalized_inputs], columns=features)

st.sidebar.markdown("---")
st.sidebar.subheader("Normalized Formulation (Strict 100%)")
st.sidebar.dataframe(input_df.T.rename(columns={0: "Actual wt%"}).style.format("{:.2f}%"))

# --- 4. Predictions & Efficacy Scoring ---
preds = {target: models[target].predict(input_df)[0] for target in targets}

ideal_size = 150.0 
size_penalty = abs(preds['droplet_size_nm'] - ideal_size) * 0.5
efficacy_score = max(0, (preds['spf'] * 2) - size_penalty)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted SPF", f"{preds['spf']:.2f}")
col2.metric("Droplet Size", f"{preds['droplet_size_nm']:.1f} nm")
col3.metric("Viscosity", f"{preds['viscosity_cp']:.1f} cP")
col4.metric("Overall Efficacy Score", f"{efficacy_score:.2f}")

st.markdown("---")

# --- 5. Visualizations ---
st.subheader("Candidate Efficacy Profile & Diagnostics")

viz_col1, viz_col2, viz_col3 = st.columns(3)

# 5A. Radar Chart
with viz_col1:
    st.markdown("**Efficacy Balance**")
    max_spf, max_size, max_visc = 50.0, 300.0, 100.0
    radar_spf = min(100, (preds['spf'] / max_spf) * 100)
    radar_size = max(0, 100 - ((preds['droplet_size_nm'] / max_size) * 100))
    radar_visc = min(100, (preds['viscosity_cp'] / max_visc) * 100)
    radar_score = min(100, efficacy_score)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[radar_spf, radar_size, radar_visc, radar_score, radar_spf],
        theta=['SPF', 'Size (Inverted)', 'Viscosity', 'Score', 'SPF'],
        fill='toself', name='Candidate', line_color='blue'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_radar, use_container_width=True)

# 5B. Feature Importance
with viz_col2:
    st.markdown("**Feature Importance (Droplet Size)**")
    xgb_model = models['droplet_size_nm'].named_steps['regressor']
    importance_df = pd.DataFrame({
        'Ingredient': ['HT', 'OL', 'Oil', 'Tween', 'Span', 'Water'],
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importance_df, x='Importance', y='Ingredient', orientation='h')
    fig_imp.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

# 5C. Response Surface
with viz_col3:
    st.markdown("**Contour (Oil vs Tween 80)**")
    grid_res = 20
    ipm_range = np.linspace(10.0, 30.0, grid_res)
    t80_range = np.linspace(2.0, 10.0, grid_res)
    grid_ipm, grid_t80 = np.meshgrid(ipm_range, t80_range)
    
    flat_ipm = grid_ipm.ravel()
    flat_t80 = grid_t80.ravel()
    
    grid_df = pd.DataFrame({
        'hydroxytyrosol_wt_pct': np.full(flat_ipm.shape, normalized_inputs[0]),
        'oleuropein_wt_pct': np.full(flat_ipm.shape, normalized_inputs[1]),
        'isopropyl_myristate_wt_pct': flat_ipm,
        'tween80_wt_pct': flat_t80,
        'span80_wt_pct': np.full(flat_ipm.shape, normalized_inputs[4]),
        'aqueous_phase_wt_pct': np.full(flat_ipm.shape, normalized_inputs[5])
    })
    
    grid_preds = models['droplet_size_nm'].predict(grid_df).reshape(grid_res, grid_res)
    
    fig_contour = go.Figure(data=go.Contour(
        z=grid_preds, x=ipm_range, y=t80_range, colorscale='Viridis', colorbar=dict(title='nm')
    ))
    fig_contour.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_contour, use_container_width=True)

st.markdown("---")

# --- 6. Consensus-Driven Multi-Objective Active Learning ---
st.subheader("Phase 3: Ensemble Optimization & Consensus Node")
st.write("Running a multi-algorithm ensemble (Bayesian + Global Scavenger) with K-Means clustering to extract the top 5 most diverse, high-efficacy, and chemically valid formulations.")

if st.button("Generate Next 5 Optimal Formulations"):
    with st.spinner("Initializing Ensemble Nodes (Memory-Safe Mode)..."):
        try:
            import torch
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from botorch.fit import fit_gpytorch_mll
            from torch.distributions import Normal
            from sklearn.cluster import KMeans
            
            # 1. Load Data
            # MAKE SURE THE FILENAME BELOW MATCHES YOUR ACTUAL GITHUB FILE
            hist_df = pd.read_csv('synthetic_formulation_data.csv') 
            X_train = torch.tensor(hist_df[features].values, dtype=torch.double)
            spf = torch.tensor(hist_df['spf'].values, dtype=torch.double)
            size = torch.tensor(hist_df['droplet_size_nm'].values, dtype=torch.double)
            
            # Target Objective for GP
            Y_train = (spf - (size * 0.1)).unsqueeze(-1)
            Y_mean, Y_std = Y_train.mean(), Y_train.std()
            Y_norm = (Y_train - Y_mean) / (Y_std + 1e-8)

            # 2. Train Bayesian Node (Gaussian Process)
            gp = SingleTaskGP(X_train, Y_norm)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # 3. Generate Candidate Pool (Memory Safe Limit: 10,000)
            n_samples = 10000
            cand_ht = torch.empty(n_samples).uniform_(0.1, 5.0)
            cand_ol = torch.empty(n_samples).uniform_(0.1, 5.0)
            cand_ipm = torch.empty(n_samples).uniform_(10.0, 30.0)
            cand_t80 = torch.empty(n_samples).uniform_(2.0, 10.0)
            cand_s80 = torch.empty(n_samples).uniform_(1.0, 5.0)
            cand_aq = torch.empty(n_samples).uniform_(40.0, 90.0)
            
            raw_cands = torch.stack([cand_ht, cand_ol, cand_ipm, cand_t80, cand_s80, cand_aq], dim=1)
            row_sums = raw_cands.sum(dim=1, keepdim=True)
            norm_cands = (raw_cands / row_sums) * 100.0

            # 4. Apply Hard Chemistry Constraints (Physics Filter)
            t80_col = norm_cands[:, 3]
            s80_col = norm_cands[:, 4]
            total_surf_col = t80_col + s80_col
            hlb_col = ((t80_col * 15.0) + (s80_col * 4.3)) / total_surf_col
            
            valid_mask = (total_surf_col <= 10.0) & (hlb_col >= 8.0) & (hlb_col <= 18.0)
            valid_cands = norm_cands[valid_mask]
            
            if len(valid_cands) < 5:
                st.error("Constraints are too tight. Not enough physically valid combinations found. Try generating again.")
            else:
                # 5. Node 1: Evaluate Expected Improvement (Bayesian Uncertainty) - Batched to save RAM
                best_f = Y_norm.max()
                gp.eval()
                with torch.no_grad():
                    # Evaluate GP only on a maximum of 3000 valid candidates to prevent OOM
                    eval_cands = valid_cands[:3000] 
                    posterior = gp(eval_cands.double())
                    mean = posterior.mean
                    sigma = posterior.variance.clamp_min(1e-9).sqrt()
                    u = (mean - best_f) / sigma
                    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
                    ucdf = normal.cdf(u)
                    updf = normal.log_prob(u).exp()
                    ei_scores = sigma * (u * ucdf + updf)
                
                # 6. Node 2: Evaluate Direct Surrogate Efficacy
                eval_cands_np = eval_cands.numpy()
                eval_df = pd.DataFrame(eval_cands_np, columns=features)
                
                preds_spf = models['spf'].predict(eval_df)
                preds_size = models['droplet_size_nm'].predict(eval_df)
                
                size_penalty = np.abs(preds_size - 150.0) * 0.5
                efficacy_scores = np.maximum(0, (preds_spf * 2) - size_penalty)
                
                # Normalize both scores to combine them equally (Consensus Logic)
                ei_norm = (ei_scores.squeeze().numpy() - np.min(ei_scores.numpy())) / (np.ptp(ei_scores.numpy()) + 1e-8)
                eff_norm = (efficacy_scores - np.min(efficacy_scores)) / (np.ptp(efficacy_scores) + 1e-8)
                consensus_scores = (ei_norm * 0.5) + (eff_norm * 0.5)
                
                # Select top 50 candidates based on Consensus Score
                top_50_idx = np.argsort(consensus_scores)[-50:]
                top_50_cands = eval_cands_np[top_50_idx]
                top_50_scores = consensus_scores[top_50_idx]
                
                # 7. Consensus Node (Diversity Clustering via K-Means)
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(top_50_cands)
                
                final_5_idx = []
                for cluster_id in range(5):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    best_in_cluster = cluster_indices[np.argmax(top_50_scores[cluster_indices])]
                    final_5_idx.append(best_in_cluster)
                
                best_formulations = top_50_cands[final_5_idx]
                next_batch_df = pd.DataFrame(best_formulations, columns=features)
                
                st.success("Ensemble Optimization Complete! Here are 5 strictly diverse, high-confidence O/W formulations:")
                st.dataframe(next_batch_df.style.format("{:.2f}"))
                
        except Exception as e:
            st.error(f"Ensemble Optimization failed: {e}")


