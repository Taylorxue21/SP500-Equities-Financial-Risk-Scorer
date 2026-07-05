# 📈 US Equities Financial Risk Scorer (US-Risk-Scorer)

<p align="left">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/Model-XGBoost-orange.svg" alt="XGBoost"></a>
  <a href="https://shap.readthedocs.io/"><img src="https://img.shields.io/badge/XAI-SHAP-red.svg" alt="SHAP"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

An XGBoost-powered, fully explainable financial risk analysis system for US public companies. It integrates multi-dimensional market data, fundamental financial statements, and macroeconomic indicators into a buy-side grade Streamlit dashboard. 

**Target**: Predict if a company will severely underperform the S&P 500 (`Alpha < -30%`) over the next 12 months, and explain *why* using SHAP local attribution.

## ✨ Core Features

- **🚨 Top-Down Risk Leaderboard**: Instantly screens the market universe to flag companies with the highest probability of severe drawdown, complete with auto-generated risk drivers.
- **🌌 Multi-Dimensional Risk Landscape**: Dynamic parallel coordinates to map out structural risks across sectors. Visually identify financial anomalies in high-leverage or low-cash environments.
- **🕸️ Sector Peer Radar**: A smart-theme adapted polar chart that benchmarks a specific company against its GICS sector median baseline in real-time.
- **📉 Explainable AI (XAI) Attribution**: A highly customized, presentation-ready SHAP waterfall chart. It breaks down the "black-box" XGBoost prediction into granular financial impacts (Log-Odds), providing absolute transparency.
- **☁️ Zero-Cost Deployment**: Designed to run entirely on free-tier services (e.g., local PostgreSQL + Streamlit) for automated, cost-free monitoring.

## 📸 Dashboard Showcase

| Macro View: Risk Landscape | Micro View: Peer Comparison | Explainability: SHAP Attribution |
| :---: | :---: | :---: |
| <img src="docs/images/parallel_coordinates.png" width="300" /> | <img src="docs/images/radar_chart.png" width="300" /> | <img src="docs/images/shap_waterfall.png" width="300" /> |

## 🚀 Quick Start (Local Deployment)

### 1. Prerequisites
Ensure you have PostgreSQL installed and running. Create a `.env` file in the root directory:
```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=postgres
```

###2. Installation
Clone the repository and install the required dependencies:

Bash
git clone https://github.com/yourusername/us-public-company-financial-risk-scoring.git
cd us-public-company-financial-risk-scoring
pip install -r requirements.txt


###3. Run the Dashboard
Launch the interactive Streamlit application:

Bash
streamlit run app/streamlit_sector_peer_app.py
☁️ Zero-Cost Automated Deployment (Recommended)
To run this dashboard 24/7 without local computing costs, you can deploy it directly to Streamlit Community Cloud:

Push this repository to your GitHub.

Sign in to Streamlit Share.

Click New app, select this repository, and set the main file path to app/streamlit_sector_peer_app.py.

Add your database secrets in the Streamlit Advanced Settings.

Click Deploy to get your public dashboard URL instantly.

## 🧠 Modeling Approach

### 1. Robust Time-Based Evaluation Schema
To strictly prevent look-ahead bias and data leakage, the dataset is split using a chronological spine. The model is trained on pre-2020 data, tuned on 2021-2022 macroeconomic shocks, and evaluated blindly on the 2023-2024 high-interest-rate regime.

### 2. Tail-Risk Decision Boundary Optimization
Traditional classifiers default to a `0.5` probability threshold, which is commercially fatal for tail-risk management. As visualized in our **Precision-Recall Curve** below, the asset's downside risk is highly non-linear. By maximizing the F1-Score on the validation surface, the system discovers an optimal risk trigger at **`0.0839`**.

| Precision-Recall Curve (AUC = 0.959) | Optimized Wind-Tunnel Confusion Matrix |
| :---: | :---: |
| <img src="docs/images/pr_curve.png" width="380" /> | <img src="docs/images/confusion_matrix_optimal.png" width="350" /> |

* **Downside Protection (Recall = 97.03%)**: At the `0.0839` systemic threshold, the model successfully captures and intercepts 97% of the absolute drawdowns (`Alpha <= -30%`) in the blind test set.
* **Signal Reliability (Precision = 85.45%)**: Despite the aggressive protective posture, the signal maintains an 85.4% accuracy rate, significantly minimizing false alarms and preserving portfolio opportunity costs.

### 3. Zero-Latency Inference Architecture
Engineered a decoupled production pipeline. By implementing `@st.cache_resource` for offline model loading, the system bypasses real-time training overhead, delivering instant buy-side risk scoring capabilities without latency.

### 4. Explainability (XAI)
Utilizes `shap.TreeExplainer` to decompose every prediction into readable financial metrics, breaking the traditional machine learning black-box and aligning with institutional compliance standards.

---

## 🗄️ Project Structure

```text
├── app/
│   └── streamlit_sector_peer_app.py   # Main Streamlit dashboard
├── docs/
│   └── images/                        # High-res architecture & chart screenshots
├── models/
│   ├── xgb_risk_model.json            # Pre-trained XGBoost weights
│   └── model_features.pkl             # Feature schema for inference
├── notebooks/
│   └── 03_modeling_xgboost.ipynb      # Offline research, PR tuning, and training
├── sql/                               # PostgreSQL schema and data pipelines
├── .env.example                       # Environment variables template
├── requirements.txt                   # Python dependencies
└── README.md
```

⚠️ Disclaimer
This project is for educational and research purposes only. The risk scores, SHAP explanations, and market analyses generated by this system do not constitute financial, investment, or trading advice. Financial markets are inherently risky, and historical data modeling cannot guarantee future performance. Please consult a certified financial advisor before making any investment decisions.

Author: Yanwen Xue | Undergraduate Student in Statistics, Korea University
