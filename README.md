<div align="right">
  🌐 <b>Language:</b>
  English | <a href="README_kr.md">한국어</a> | <a href="README_zh.md">简体中文</a>
</div>

# 📈 US Equities Financial Risk Scorer (US-Risk-Scorer)

<p align="left">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/Model-XGBoost-orange.svg" alt="XGBoost"></a>
  <a href="https://shap.readthedocs.io/"><img src="https://img.shields.io/badge/XAI-SHAP-red.svg" alt="SHAP"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

> This is an XGBoost-based interpretable financial risk analysis system targeting US listed companies. The system integrates multi-dimensional market data, fundamental financial statements, and macroeconomic indicators, and builds an interactive Streamlit analytical dashboard to buy-side standards.

> **Objective**: Predict whether a company will significantly underperform the S&P 500 (`Alpha < -30%`) over the next 12 months, and explain the reasoning through SHAP local attribution.

## 1. Business Background and Problem Definition

**Business Problem:**
Can we identify US listed companies at risk of significantly underperforming the broader market over the next year, using only public financial statements, market behavior data, and macroeconomic indicators?

**Target Variable Definition:**

(`risk_label = 1`) If the relative return over the next 12 months (relative to SPY) <= -30% Otherwise (`risk_label = 0`)  


This learning project is a **business decision-oriented data analysis practice**: the core value lies not in the model itself, but in:

- How to transform raw financial data into **interpretable and comparable** analytical metrics (e.g., year-over-year growth rate, industry relative quantiles, leverage/cash structure);
- How to use **statistical and visualization methods** to verify whether features are genuinely correlated with risk;
- How to **translate model outputs into business language** (e.g., risk scores, industry comparisons, driving factors)

---

## 2. Analytics Workflow

```
Data Acquisition Layer          →  Financial Statements / Historical Market Data / Macro Indicators (FMP API + FRED API)
↓
Data Cleaning & Modeling Layer  →  PostgreSQL Storage + SQL Feature Engineering (Avoid Look-ahead Bias / Strictly Align by Report Date)
↓
Exploratory Analysis Layer      →  Industry Distribution Boxplots / Time Series Label Rates / Correlation & Quantile Analysis
↓
Modeling & Validation Layer     →  Logistic Regression Baseline vs. XGBoost Main Model, Time Series Split, PR-AUC / Precision / Top-K Hit Rate
↓
Interpretability Layer          →  SHAP Global Feature Importance + Single Company Local Attribution (Waterfall Chart)
↓
Business Presentation Layer     →  Streamlit Interactive Dashboard: Risk Leaderboard / Industry Comparison / Radar Chart / Attribution Explanation
```

---

## 3. Data Description

| Dimension | Description |
|---|---|
| Coverage | US listed companies (prioritizing the S&P 500 subset, 100–200 highly liquid tickers) |
| Observation Frequency | Quarterly financial statements + Aggregated daily market data |
| Time Span | 2015–2025 |
| Benchmark Index | SPY (S&P 500 ETF) |
| Data Sources | Financial Modeling Prep (Financials and Market Data), FRED (Macroeconomic Indicators) |

### Feature System (4 Major Categories, 20+ Metrics)

- **Profitability & Growth**: Gross margin, operating margin, net margin, ROE, ROA, YoY revenue growth
- **Solvency & Liquidity**: Current ratio, cash ratio, `debt_to_assets`, `cash_to_assets`
- **Cash Flow Quality**: Free cash flow margin, operating cash flow/debt (`ocf_to_debt`), negative FCF flag
- **Market Behavior**: 1M/3M/6M/12M returns, volatility, maximum drawdown, relative excess return to SPY
- **Macroeconomic Environment**: Federal funds rate, inflation rate, unemployment rate, 10Y-2Y yield spread
- **Industry Relative Features**: Intra-industry leverage quantiles, ROE quantiles, etc. (eliminating systematic differences within industries)

> **Preventing Look-ahead Bias** — All features strictly use information available on or before the `as_of_date`, with financial statement data aligned using conservative lagging.
---

## 4. Modeling & Evaluation

Adopted **time series splitting** rather than random splitting, which is closer to real business scenarios (using history to predict the future, instead of "data-leaking validation with future data"):

Training Set: 2015–2020
Validation Set: 2021–2022
Test Set: 2023–2025


| Model | Purpose |
|---|---|
| Logistic Regression | Baseline model, verifies the linear separability of features, and facilitates business interpretation of coefficient directions. |
| XGBoost (`scale_pos_weight` to handle class imbalance) | Main model, captures non-linear interactions (e.g., combined risk of "high leverage × high volatility"). |

### Tail Risk Decision Threshold Optimization

Traditional classifiers typically default to a probability threshold of `0.5`, but in tail risk management, this setting is commercially unacceptable. As shown in our **Precision-Recall Curve** below, the downside risk of the asset exhibits significant non-linear characteristics. By maximizing the F1-Score on the validation set, the system ultimately identified the optimal risk trigger threshold as **`0.0839`**.

| Precision-Recall Curve (AUC = 0.959) | Optimized "Wind Tunnel Test" Confusion Matrix |
| :---: | :---: |
| <img src="docs/images/pr_curve.png" width="500" /> | <img src="docs/images/confusion_matrix_optimal.png" width="500" /> |

* **Downside Protection Capability (Recall = 97.03%)**: At the system-level threshold of `0.0839`, the model successfully captured and intercepted 97% of absolute drawdown samples (Alpha ≤ -30%) in the blind test dataset.

* **Signal Reliability (Precision = 85.45%)**: While maintaining an aggressive risk defense strategy, the signal still maintained an 85.4% precision rate, thereby significantly reducing false alarms and effectively controlling the opportunity cost of the portfolio.

---

## 5. Interpretability Analysis (SHAP)

The model is not delivered as a black box, but comes with comprehensive attribution analysis:

- **Global Level**: SHAP summary plot / bar plot, identifying features contributing the most to overall risk prediction (e.g., combined signals of high leverage, low cash buffer, and high volatility).
- **Stock Level**: For any selected company in the dashboard, generates a SHAP waterfall plot, showing item-by-item which financial/market features pushed this company towards "high risk" or "low risk".

This layer is the key to the entire project moving **from a "model" to a "decision tool"** — risk control or investment personnel do not need to understand XGBoost principles, they only need to understand "why this company was flagged as high risk".

<p align="center">
  <img src="docs/images/shap_waterfall.png" width="85%" alt="SHAP Local Waterfall Explanation">
  <br>
  <i style="color: gray; font-size: 14px;">Figure: SHAP individual stock risk attribution, breaking down the specific contributions of various financial and macro indicators to the final risk score</i>
</p>


## 6. Interactive Analysis Dashboard (Streamlit Dashboard)

The dashboard is designed around the analytical logic of "**Macro → Meso (Industry) → Micro (Individual Stock)**":

1. **Top 10 High-Risk Companies Watchlist**: Ranked by market-wide risk scores (0–100 relative quantiles) + main risk-driving labels
2. **Industry Risk Distribution (Box Plots)**: Observes the "natural boundaries" of different GICS industries in leverage, cash, and volatility (e.g., the leverage center of the utility sector is naturally higher)
3. **Multi-dimensional Risk Panorama (Parallel Coordinates Plot)**: Interactive filtering to observe clustering features of high-leverage companies across multiple dimensions
4. **Individual Stock vs. Industry Peers Comparison**: Differences between the selected company's core metrics and the industry average (expressed in percentage points / multiples), overlaid with a radar chart to show market-wide quantile rankings
5. **SHAP Individual Stock Risk Attribution**: Visually explains how a single company's risk score is "assembled" by specific financial indicators

**Key Technical Implementations:**
- PostgreSQL as the data layer, connected via `SQLAlchemy`, with local CSV fallback loading (ensuring the stability of the online Demo and avoiding blank pages caused by database connection failures)
- `Plotly` implements interactive box plots / parallel coordinates plots / radar charts; `Matplotlib` hosts SHAP static plots
- Caching strategy: `st.cache_data` / `st.cache_resource` separates data loading and model training, avoiding repeated calculations for every interaction

<p align="center">
  <img src="docs/images/radar_chart.png" width="45%" alt="Company vs Sector Radar Chart">
  <img src="docs/images/parallel_coordinates.png" width="45%" alt="Multi-dimensional Risk Parallel Coordinates">
  <br>
  <i style="color: gray; font-size: 14px;">Figure: Individual Stock vs. Industry Peers Comparison (Left) and Multi-dimensional Risk Panorama (Right)</i>
</p>


## 7. Tech Stack

| Category | Tools |
|---|---|
| Data Acquisition | Financial Modeling Prep API, FRED API |
| Data Storage & Modeling | PostgreSQL, SQL (Feature engineering is fully pushed down to the SQL layer, rather than piled up in Python) |
| Data Processing | Python (pandas, numpy) |
| Modeling | scikit-learn (Logistic Regression), XGBoost |
| Interpretability | SHAP |
| Visualization & Dashboard | Streamlit, Plotly, Matplotlib |
| Engineering | dotenv for environment variable management, modular scripts (`fetch_*` / `clean_*` / `build_*` / `train_*`) |

---

## 8. Project Structure
```text
us-public-company-financial-risk-scoring/
├── .vscode/
│   └── settings.json
├── app/
│   └── streamlit_sector_peer_app.py          
├── data/
│   ├── processed/                             # Feature storage (datasets directly usable by the model)
│   │   ├── model_dataset.parquet
│   │   └── test_scored.parquet
│   └── raw/                                   
│       ├── fmp_income_statements.csv
│       └── yfinance_income_statements.csv
├── models/                                    
│   ├── model_features.pkl
│   ├── xgb_risk_model.json
│   └── xgboost_risk_model.pkl
├── notebooks/                                  
│   ├── 01_data_quality_check.ipynb
│   ├── 02_export_dataset.ipynb
│   └── 03_modeling_xgboost.ipynb              # Trained XGBoost model (exported for production environment)
├── reports/                                   
│   ├── feature_importance.png
│   ├── risk_leaderboard.csv
│   ├── risk_report.py
│   └── top_features.csv
├── sql/                                        # Database feature engineering scripts
│   ├── 03_build_financial_features.sql
│   ├── 04_build_risk_labels.sql
│   └── 06_build_model_dataset.sql
├── src/                                        
│   ├── build_dataset.py
│   ├── build_features.py
│   ├── build_labels.py
│   ├── explain.py
│   ├── fetch_macro.py
│   ├── fetch_prices.py
│   ├── fetch_sec_ultimate.py
│   ├── fetch_sp500_sectors.py
│   ├── fetch_spy.py
│   └── train.py
├── .env                                        
├── .env.example                              
├── dataset.csv                              
├── requirements.txt                           
├── README.md                                   
├── README_zh.md                                
└── README_kr.md
```

## 9. How to Run

```Bash
# 1. Clone the repository and install dependencies
git clone <repo-url>
cd us-public-company-financial-risk-scoring
pip install -r requirements.txt

# 2. Configure environment variables (API Keys, local paths, etc.)
cp .env.example .env
# Note: Please edit the .env file locally to configure your API keys

# 3. Run the end-to-end data acquisition pipeline (API)
python src/fetch_sp500_sectors.py
python src/fetch_sec_ultimate.py
python src/fetch_prices.py
python src/fetch_macro.py

# 4. Build feature and label matrices (local storage or PostgreSQL pushdown)
# If building using SQL, execute the scripts in the sql/ directory sequentially in the database
python src/build_features.py
python src/build_labels.py
python src/build_dataset.py

# 5. Model training, evaluation, and SHAP attribution export
python src/train.py
python src/explain.py

# 6. Launch the Streamlit interactive dashboard
streamlit run app/streamlit_sector_peer_app.py
```

---

## 10. Limitations and Future Plans

**Current Limitations:**
- The sample size is concentrated on highly liquid large-cap stocks; generalization to small-cap / newly listed stocks has not been verified.
- Financial data is subject to reporting lags and metric adjustments; cross-company comparisons have not been corrected for differences in industry accounting standards.
- Textual information (e.g., earnings call sentiment, news sentiment) is not included; risk signals currently rely solely on structured data.
---

## ⚠️ Disclaimer

This project is for educational and research purposes only.

The risk scores, feature attributions (SHAP), and all analysis results generated by the system are solely used to demonstrate data-driven financial modeling methods and do not constitute any investment, trading, or financial advice.

Financial markets are highly uncertain. Models built on historical data have inherent limitations and cannot guarantee accurate predictions of future market performance.

Users should not rely on this system to make any actual investment decisions and should consult a professional financial advisor before making any finance-related decisions.

---

# 👨‍💻 Author

**Xue Yanwen**  
B.S. in Statistics, Korea University

