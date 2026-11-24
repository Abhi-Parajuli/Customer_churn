# 📡 Churn Radar — Telecom Customer Churn Prediction & Retention Targeting

A Streamlit application that predicts which telecom customers are likely to churn, explains what drives churn, and decides **who should receive a retention offer** using a cost-based threshold instead of the default 0.5.

Built on the IBM Telco Customer Churn dataset (7,043 customers, 21 columns; 26.6% churn).

<!-- Add a screenshot after deploying: ![Churn Radar](assets/screenshot.png) -->

---

## ✨ Features

### 🔮 Predict
- Interactive customer profile form (account, billing, household, services)
- Churn probability from a tuned Gradient Boosting pipeline, shown on a colour-banded gauge
- Retention decision against a cost-based threshold, plus the expected net value of making an offer
- Comparison with similar customers (same contract and internet service)
- "What would move this customer's risk?" — re-scores the same customer under single changes (longer contract, auto-pay, tech support, lower bill)

### 📊 Dashboard
- KPI cards: customers, churn rate, average monthly charges (churners vs. stayers), monthly revenue lost, average tenure
- Filters by contract and internet service
- Churn rate by contract, internet service, payment method, tenure band, and protection add-ons
- **Monthly charges**: churner vs. stayer distribution, churn rate by charge band, monthly charges vs. tenure scatter, average charges by contract
- Contract × internet service churn heatmap

### 🧪 Model & cost
- Held-out test metrics with bootstrap 95% confidence intervals
- Live cost curve: change offer cost, churn loss, and success rate in the sidebar and the optimal threshold is re-tuned from out-of-fold predictions
- Confusion matrix, model comparison (5-fold CV), permutation importance, calibration curve

---

## 📈 Results (held-out test set, 1,407 customers)

| Metric | Value |
|---|---|
| ROC-AUC | 0.841 (95% CI 0.820–0.862) |
| PR-AUC | 0.658 (95% CI 0.608–0.707) |
| Operating threshold | 0.33 (chosen from the cost model on out-of-fold training predictions) |
| Churn recall / precision | 0.74 / 0.55 |
| Cost saving vs. flagging nobody | 12.6% (95% CI 10.8–14.3%) |

Main drivers: tenure, internet service type, contract type, total charges.

> **The cost figures are assumptions** (offer $50, loss $400 per churner, 40% offer success). The dollar saving illustrates the method; it is not a forecast. Estimating these values from real campaign data is the highest-value next step.

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| Frontend / UI | Streamlit, custom CSS (Fraunces + Manrope) |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn 1.8.0, joblib |
| Visualization | Plotly |
| Notebook analysis | SciPy, Matplotlib, Seaborn, SHAP |
| DevOps | Dev Containers (`.devcontainer/devcontainer.json`) |

---

## 🏗 Architecture

```
.
├── .devcontainer/devcontainer.json
├── .streamlit/config.toml        # light theme
├── Notebook/telecom_churn_improved.ipynb
├── data/telco_churn.csv
├── models/
│   ├── model_v1.joblib           # full pipeline (preprocessing + Gradient Boosting)
│   ├── feature_columns_v1.joblib # input schema
│   ├── model_meta_v1.json        # test metrics, CIs, importance, calibration, costs
│   └── oof_predictions_v1.csv    # out-of-fold predictions for live threshold tuning
├── App.py
├── requirements.txt
└── requirements-notebook.txt
```

1. **Training (notebook)**: cleaning, tests matched to feature type (chi-square + Cramér's V, Mann-Whitney U), 8 classifiers compared with stratified 5-fold CV, randomised search on the top two, threshold chosen from out-of-fold predictions with a retention-cost model, single evaluation on the test set, then explainability and a sensitivity check.
2. **Artifacts**: the last notebook cell saves the pipeline and everything the app needs to display model results.
3. **App**: `App.py` loads the pipeline and applies the same cleaning rules as the notebook to the CSV for the dashboard.

---

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run App.py
```

To re-run the analysis: `pip install -r requirements-notebook.txt`, open the notebook from inside `Notebook/`, and run all cells. The final cell regenerates the files in `models/`.

> The saved model requires **scikit-learn 1.8.0**. Other versions may fail to load it or warn about version mismatch.

---

## ⚠️ Limitations

- Cost assumptions are illustrative, not measured.
- Single data snapshot: no out-of-time validation or drift monitoring.
- Observational data: drivers and what-if effects are associations, not causal effects.
- The model is trained on the training split only (80%); it is not refit on all data.

## 🔭 Next steps

- Estimate offer success rate and churn loss from real campaign / CLV data.
- Uplift modelling to target customers who respond to an offer, not just those likely to churn.
- Out-of-time validation and drift monitoring if deployed.
