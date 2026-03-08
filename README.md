# AI-Guided Nanoemulsion Formulation Portal

This repository contains the interactive Active Learning and Surrogate Modeling portal for the optimization of a Nano-Enabled Olive Phenolic Sunscreen.

## System Components
* **Surrogate Models (XGBoost):** Predicts Sun Protection Factor (SPF), Droplet Size (nm), and Viscosity (cP) based on ingredient mass fractions.
* **Active Learning Loop (Bayesian Optimization):** Navigates the formulation design space using a Gaussian Process surrogate and Expected Improvement (EI) acquisition function to suggest the next 5 optimal laboratory experiments.

## Repository Structure
* `app.py`: Main Streamlit application script.
* `requirements.txt`: Dependency list for cloud deployment.
* `macroscopic_surrogate_models.pkl`: Pre-trained XGBoost regression pipelines.
* `synthetic_formulation_data.csv`: Historical dataset required to initialize the Bayesian Optimization loop.

## Deployment Instructions (Streamlit Community Cloud)
1. Fork or upload these files to a public/private GitHub repository.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select the repository, set the branch to `main`, and the main file path to `app.py`.
5. Click **Deploy**.

## Updating Models with Real Data
When the laboratory team provides real wet-lab results:
1. Run the local retraining script (`retrain_pipeline.py`).
2. Replace the old `macroscopic_surrogate_models.pkl` and `synthetic_formulation_data.csv` files in this repository with the newly generated ones.
3. Commit and push the changes to GitHub. The Streamlit portal will automatically restart and serve the updated models.