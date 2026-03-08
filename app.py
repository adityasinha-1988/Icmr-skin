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

# --- 6. Bayesian Optimization Loop ---
st.subheader("Phase 3: Automated Active Learning")
st.write("Clicking below initializes the Bayesian models to suggest the next 5 optimal formulations.")

if st.button("Generate Next 5 Optimal Formulations"):
    with st.spinner("Initializing ML Libraries & Running Bayesian Optimization..."):
        try:
            # LAZY IMPORT
            import torch
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from botorch.fit import fit_gpytorch_mll
            from botorch.acquisition import qExpectedImprovement
            from botorch.optim import optimize_acqf
            from botorch.utils.transforms import normalize, unnormalize

            hist_df = pd.read_csv('synthetic_formulation_data.csv')
            X = torch.tensor(hist_df[features].values, dtype=torch.double)
            
            spf = torch.tensor(hist_df['spf'].values, dtype=torch.double)
            size = torch.tensor(hist_df['droplet_size_nm'].values, dtype=torch.double)
            Y = (spf - (size * 0.1)).unsqueeze(-1)

            bounds = torch.tensor([
                [0.1, 0.1, 10.0, 2.0, 1.0, 40.0],
                [5.0, 5.0, 30.0, 10.0, 5.0, 90.0]
            ], dtype=torch.double)
            
            X_norm = normalize(X, bounds)
            
            gp = SingleTaskGP(X_norm, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            acq_func = qExpectedImprovement(model=gp, best_f=Y.max())
            
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double),
                q=5, num_restarts=5, raw_samples=20,
            )
            
            real_candidates = unnormalize(candidates, bounds)
            row_sums = real_candidates.sum(dim=1, keepdim=True)
            normalized_candidates = (real_candidates / row_sums) * 100
            
            next_batch_df = pd.DataFrame(normalized_candidates.detach().numpy(), columns=features)
            
            st.success("Optimization Complete! Give these compositions to the lab team:")
            st.dataframe(next_batch_df.style.format("{:.2f}"))
            
        except FileNotFoundError:
            st.error("'synthetic_formulation_data.csv' nahi mili. Please ensure this file is uploaded to your GitHub repository.")
        except Exception as e:
            st.error(f"Optimization failed: {e}")

