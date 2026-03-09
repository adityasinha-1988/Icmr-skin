import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings
import io

warnings.filterwarnings("ignore")

# --- 1. System Setup & Strict Session State ---
st.set_page_config(page_title="Nano-Sunscreen Optimizer", layout="wide")

if 'ht_val' not in st.session_state: st.session_state.ht_val = 1.0
if 'ol_val' not in st.session_state: st.session_state.ol_val = 2.0
if 'ipm_val' not in st.session_state: st.session_state.ipm_val = 20.0
if 't80_val' not in st.session_state: st.session_state.t80_val = 5.0
if 's80_val' not in st.session_state: st.session_state.s80_val = 2.5
if 'aq_val' not in st.session_state: st.session_state.aq_val = 69.5

def apply_formulation():
    if 'generated_df' in st.session_state and 'selected_formulation_idx' in st.session_state:
        idx = st.session_state.selected_formulation_idx
        row = st.session_state['generated_df'].iloc[idx]
        
        st.session_state.ht_val = float(np.clip(row['hydroxytyrosol_wt_pct'], 0.1, 5.0))
        st.session_state.ol_val = float(np.clip(row['oleuropein_wt_pct'], 0.1, 5.0))
        st.session_state.ipm_val = float(np.clip(row['isopropyl_myristate_wt_pct'], 10.0, 30.0))
        st.session_state.t80_val = float(np.clip(row['tween80_wt_pct'], 2.0, 10.0))
        st.session_state.s80_val = float(np.clip(row['span80_wt_pct'], 1.0, 5.0))
        st.session_state.aq_val = float(np.clip(row['aqueous_phase_wt_pct'], 40.0, 90.0))

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

features = [
    'hydroxytyrosol_wt_pct', 'oleuropein_wt_pct', 
    'isopropyl_myristate_wt_pct', 'tween80_wt_pct', 
    'span80_wt_pct', 'aqueous_phase_wt_pct'
]
targets = ['spf', 'droplet_size_nm', 'viscosity_cp']

# --- 2. Bio-Team Data Integration ---
st.sidebar.header("🧪 Lab Data Integration")

metadata_cols = ['experiment_id', 'date', 'time']
template_df = pd.DataFrame(columns=metadata_cols + features + targets)
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Blank CSV Template", 
    data=csv_template, 
    file_name="lab_data_template.csv", 
    mime="text/csv"
)

uploaded_file = st.sidebar.file_uploader("Upload Wet-Lab Results (.csv)", type=["csv"], key="lab_data_upload")

if uploaded_file is not None:
    st.sidebar.success(f"File loaded: {uploaded_file.name}")
    
    if st.sidebar.button("Retrain AI Models Now"):
        with st.spinner("Retraining XGBoost (Single-Core Mode)..."):
            try:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from xgboost import XGBRegressor
                
                uploaded_file.seek(0) 
                lab_df = pd.read_csv(uploaded_file)
                
                if not all(col in lab_df.columns for col in features + targets):
                    st.sidebar.error("CSV columns mismatch.")
                else:
                    X_lab = lab_df[features]
                    new_models = {}
                    for target in targets:
                        y_lab = lab_df[target]
                        pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('regressor', XGBRegressor(random_state=42, n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=1))
                        ])
                        pipe.fit(X_lab, y_lab)
                        new_models[target] = pipe
                    
                    buffer = io.BytesIO()
                    joblib.dump(new_models, buffer)
                    buffer.seek(0)
                    
                    st.session_state['retrained_model'] = buffer.getvalue()
                    st.session_state['retrain_success'] = True
                    
            except Exception as e:
                st.sidebar.error(f"Retraining failed: {e}")

if st.session_state.get('retrain_success', False):
    st.sidebar.success("Models retrained successfully!")
    st.sidebar.download_button(
        label="Download Updated Model (.pkl)", 
        data=st.session_state['retrained_model'], 
        file_name="macroscopic_surrogate_models_UPDATED.pkl", 
        mime="application/octet-stream"
    )

st.sidebar.markdown("---")

# --- 3. Interactive Inputs & Normalization ---
st.sidebar.header("Formulation Inputs (Raw wt%)")
ht = st.sidebar.slider("Hydroxytyrosol", 0.1, 5.0, step=0.1, key='ht_val')
ol = st.sidebar.slider("Oleuropein", 0.1, 5.0, step=0.1, key='ol_val')
ipm = st.sidebar.slider("Isopropyl Myristate (Oil)", 10.0, 30.0, step=0.5, key='ipm_val')
t80 = st.sidebar.slider("Tween 80", 2.0, 10.0, step=0.1, key='t80_val')
s80 = st.sidebar.slider("Span 80", 1.0, 5.0, step=0.1, key='s80_val')
aq = st.sidebar.slider("Aqueous Phase", 40.0, 90.0, step=0.5, key='aq_val')

raw_inputs = np.array([ht, ol, ipm, t80, s80, aq])
normalized_inputs = (raw_inputs / raw_inputs.sum()) * 100
input_df = pd.DataFrame([normalized_inputs], columns=features)

current_tween, current_span = normalized_inputs[3], normalized_inputs[4]
total_surf = current_tween + current_span
current_hlb = ((current_tween * 15.0) + (current_span * 4.3)) / total_surf if total_surf > 0 else 0

st.sidebar.markdown("---")
st.sidebar.subheader("Physicochemical Constraints")
st.sidebar.write(f"**Total Surfactant:** {total_surf:.2f}% (Safe Limit: ≤ 10%)")
st.sidebar.write(f"**HLB Value:** {current_hlb:.2f} (O/W Target: 8 - 18)")

if total_surf > 10.0: st.sidebar.error("Warning: High surfactant load.")
if not (8 <= current_hlb <= 18): st.sidebar.error("Warning: HLB is outside O/W range.")

st.sidebar.markdown("---")
st.sidebar.subheader("Normalized Formulation (100%)")
st.sidebar.dataframe(input_df.T.rename(columns={0: "Actual wt%"}).style.format("{:.2f}%"))

# --- 4. Predictions & Efficacy Scoring ---
preds = {target: models[target].predict(input_df)[0] for target in targets}
size_penalty = abs(preds['droplet_size_nm'] - 150.0) * 0.5
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

with viz_col1:
    st.markdown("**Efficacy Balance**")
    radar_spf = min(100, (preds['spf'] / 50.0) * 100)
    radar_size = max(0, 100 - ((preds['droplet_size_nm'] / 300.0) * 100))
    radar_visc = min(100, (preds['viscosity_cp'] / 100.0) * 100)
    radar_score = min(100, efficacy_score)

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[radar_spf, radar_size, radar_visc, radar_score, radar_spf],
        theta=['SPF', 'Size (Inverted)', 'Viscosity', 'Score', 'SPF'], fill='toself'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_radar, width='stretch')

with viz_col2:
    st.markdown("**Feature Importance (Size)**")
    xgb_model = models['droplet_size_nm'].named_steps['regressor']
    importance_df = pd.DataFrame({'Ingredient': ['HT', 'OL', 'Oil', 'Tween', 'Span', 'Water'], 'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=True)
    fig_imp = px.bar(importance_df, x='Importance', y='Ingredient', orientation='h')
    fig_imp.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_imp, width='stretch')

with viz_col3:
    st.markdown("**Contour (Oil vs Tween 80)**")
    ipm_range = np.linspace(10.0, 30.0, 20)
    t80_range = np.linspace(2.0, 10.0, 20)
    grid_ipm, grid_t80 = np.meshgrid(ipm_range, t80_range)
    
    grid_df = pd.DataFrame({
        'hydroxytyrosol_wt_pct': np.full(grid_ipm.ravel().shape, normalized_inputs[0]),
        'oleuropein_wt_pct': np.full(grid_ipm.ravel().shape, normalized_inputs[1]),
        'isopropyl_myristate_wt_pct': grid_ipm.ravel(),
        'tween80_wt_pct': grid_t80.ravel(),
        'span80_wt_pct': np.full(grid_ipm.ravel().shape, normalized_inputs[4]),
        'aqueous_phase_wt_pct': np.full(grid_ipm.ravel().shape, normalized_inputs[5])
    })
    grid_preds = models['droplet_size_nm'].predict(grid_df).reshape(20, 20)
    fig_contour = go.Figure(data=go.Contour(z=grid_preds, x=ipm_range, y=t80_range, colorscale='Viridis', colorbar=dict(title='nm')))
    fig_contour.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_contour, width='stretch')

st.markdown("---")

# --- 6. Consensus-Driven Multi-Objective Active Learning ---
st.subheader("Phase 3: Ensemble Optimization & Consensus Node")
st.write("Generate diverse, high-efficacy, and chemically valid formulations (HLB 8-18, Surfactant ≤ 10%).")

run_mode = st.radio("Select Optimization Engine:", ["⚡ Fast Mode (XGBoost Surrogate + Clustering - Instant)", "🔬 Deep Mode (Includes Bayesian GP - Takes ~30 seconds)"])

if st.button("Generate Next 5 Optimal Formulations"):
    with st.spinner(f"Running {run_mode.split(' ')[1]}..."):
        try:
            import torch
            from sklearn.cluster import KMeans
            
            # Increased pool size to survive strict filtering
            n_samples = 50000 
            cand_ht = torch.empty(n_samples).uniform_(0.1, 5.0)
            cand_ol = torch.empty(n_samples).uniform_(0.1, 5.0)
            cand_ipm = torch.empty(n_samples).uniform_(10.0, 30.0)
            cand_t80 = torch.empty(n_samples).uniform_(2.0, 10.0)
            cand_s80 = torch.empty(n_samples).uniform_(1.0, 5.0)
            cand_aq = torch.empty(n_samples).uniform_(40.0, 90.0)
            
            raw_cands = torch.stack([cand_ht, cand_ol, cand_ipm, cand_t80, cand_s80, cand_aq], dim=1)
            row_sums = raw_cands.sum(dim=1, keepdim=True)
            norm_cands = (raw_cands / row_sums) * 100.0

            ht_col = norm_cands[:, 0]
            ol_col = norm_cands[:, 1]
            ipm_col = norm_cands[:, 2]
            t80_col = norm_cands[:, 3]
            s80_col = norm_cands[:, 4]
            aq_col = norm_cands[:, 5]

            total_surf_col = t80_col + s80_col
            hlb_col = ((t80_col * 15.0) + (s80_col * 4.3)) / total_surf_col
            
            # STRICT BOUNDARY FILTER: Applies slider constraints directly to the normalized data
            valid_mask = (
                (total_surf_col <= 10.0) & 
                (hlb_col >= 8.0) & 
                (hlb_col <= 18.0) &
                (ht_col >= 0.1) & (ht_col <= 5.0) &
                (ol_col >= 0.1) & (ol_col <= 5.0) &
                (ipm_col >= 10.0) & (ipm_col <= 30.0) &
                (t80_col >= 2.0) & (t80_col <= 10.0) &
                (s80_col >= 1.0) & (s80_col <= 5.0) &
                (aq_col >= 40.0) & (aq_col <= 90.0)
            )
            valid_cands = norm_cands[valid_mask]
            
            if len(valid_cands) < 5:
                st.error(f"Constraints are too tight. Only {len(valid_cands)} valid combinations found. Try generating again.")
            else:
                eval_cands_np = valid_cands.numpy()
                eval_df = pd.DataFrame(eval_cands_np, columns=features)
                
                preds_spf = models['spf'].predict(eval_df)
                preds_size = models['droplet_size_nm'].predict(eval_df)
                efficacy_scores = np.maximum(0, (preds_spf * 2) - (np.abs(preds_size - 150.0) * 0.5))
                final_scores = efficacy_scores

                if "Deep" in run_mode:
                    from botorch.models import SingleTaskGP
                    from gpytorch.mlls import ExactMarginalLogLikelihood
                    from botorch.fit import fit_gpytorch_mll
                    from torch.distributions import Normal
                    
                    hist_df = pd.read_csv('synthetic_formulation_data.csv') 
                    X_train = torch.tensor(hist_df[features].values, dtype=torch.double)
                    spf = torch.tensor(hist_df['spf'].values, dtype=torch.double)
                    size = torch.tensor(hist_df['droplet_size_nm'].values, dtype=torch.double)
                    
                    Y_train = (spf - (size * 0.1)).unsqueeze(-1)
                    Y_mean, Y_std = Y_train.mean(), Y_train.std()
                    Y_norm = (Y_train - Y_mean) / (Y_std + 1e-8)

                    gp = SingleTaskGP(X_train, Y_norm)
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    fit_gpytorch_mll(mll)
                    
                    gp_cands = valid_cands[:300] 
                    best_f = Y_norm.max()
                    gp.eval()
                    with torch.no_grad():
                        posterior = gp(gp_cands.double())
                        u = (posterior.mean - best_f) / posterior.variance.clamp_min(1e-9).sqrt()
                        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
                        ei_scores = posterior.variance.clamp_min(1e-9).sqrt() * (u * normal.cdf(u) + normal.log_prob(u).exp())
                    
                    ei_norm = (ei_scores.squeeze().numpy() - np.min(ei_scores.numpy())) / (np.ptp(ei_scores.numpy()) + 1e-8)
                    eff_norm = (efficacy_scores[:300] - np.min(efficacy_scores[:300])) / (np.ptp(efficacy_scores[:300]) + 1e-8)
                    final_scores[:300] = (ei_norm * 0.5) + (eff_norm * 0.5)

                top_50_idx = np.argsort(final_scores)[-50:]
                top_50_cands = eval_cands_np[top_50_idx]
                top_50_scores = final_scores[top_50_idx]
                
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(top_50_cands)
                
                final_5_idx = [np.where(cluster_labels == i)[0][np.argmax(top_50_scores[np.where(cluster_labels == i)[0]])] for i in range(5)]
                best_formulations = top_50_cands[final_5_idx]
                
                st.session_state['generated_df'] = pd.DataFrame(best_formulations, columns=features)
                st.success("Optimization Complete!")
                
        except Exception as e:
            st.error(f"Optimization failed: {e}")

if 'generated_df' in st.session_state:
    st.dataframe(st.session_state['generated_df'].style.format("{:.2f}"))
    
    st.markdown("### Load to Dashboard")
    colA, colB = st.columns([3, 1])
    with colA:
        st.selectbox("Select a formulation from the table above to analyze its properties:", 
                     options=[0, 1, 2, 3, 4], 
                     format_func=lambda x: f"Option {x}",
                     key='selected_formulation_idx')
    with colB:
        st.write("")
        st.write("")
        st.button("Apply to Sliders", on_click=apply_formulation)
