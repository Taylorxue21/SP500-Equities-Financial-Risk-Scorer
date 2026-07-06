import os
import re
import numpy as np
import pandas as pd
import matplotlib.text as mtext
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dotenv import load_dotenv
from sqlalchemy import create_engine

st.set_page_config(page_title="US Equities Risk Scoring", page_icon="📈", layout="wide")

load_dotenv()

@st.cache_resource
def get_db_engine():
    db_user = os.getenv("DB_USER", "taylorxue")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "postgres")
    return create_engine(f"postgresql://{db_user}:{db_password}@localhost:5432/{db_name}")

@st.cache_data
def load_data():
    """
    Fallback to offline static loading to ensure zero-latency and prevent cloud instability
    """
    
    try:
        df = pd.read_csv("dataset.csv") 
    except FileNotFoundError:
        df = pd.read_csv(".../dataset.csv") 
        
    df["date"] = pd.to_datetime(df["date"])
    
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
        
    return df

def make_numeric_matrix(frame, columns=None):
    if columns is not None:
        frame = frame.reindex(columns=columns, fill_value=0)
    numeric = frame.copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float64")

def safe_float(value):
    if value is None or pd.isna(value):
        return 0.0
    return float(value)

# Model Training and Scoring Engine
@st.cache_resource
def train_model(df):
    drop_cols = ["ticker", "date", "sector", "industry", "risk_label", "alpha"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = make_numeric_matrix(X)
    y = pd.to_numeric(df["risk_label"], errors="coerce").fillna(0).astype(int)

    weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss", 
        random_state=42
    )
    model.fit(X, y)
    return model, X.columns.tolist()

# Replacement Code
# @st.cache_resource
# def load_trained_model():
#     feature_cols = joblib.load("models/model_features.pkl")
#     model = xgb.XGBClassifier()
#     model.load_model("models/xgb_risk_model.json")
#     return model, feature_cols

# model, feature_cols = load_trained_model()

def get_risk_score(prob_series):
    """
    Apply cross-sectional percentile ranking
    Force calibration of skewed probabilities into a uniform 0 to100 distribution
    """

    if not isinstance(prob_series, pd.Series):
        prob_series = pd.Series(prob_series)
        
    #rank(pct=True) returns percentile ranks in [0, 1]; multiply by 100 to convert to a 0–100 scale
    return prob_series.rank(pct=True) * 100

def get_risk_level(score):
    """
    Classify risk levels based on 0 to 100 relative risk scores 
    """
    if score <= 30:
        return "🟢 Low"      
    elif score <= 60:
        return "🟡 Moderate"  
    elif score <= 85:
        return "🟠 Elevated"  
    else:
        return "🔴 High"      

def get_latest_company_rows(df):
    return df.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()

def build_risk_leaderboard(df, model, feature_cols):
    latest_rows = get_latest_company_rows(df)
    X_latest = make_numeric_matrix(latest_rows[feature_cols], feature_cols)
    
    # Extract raw probabilities and generate business scores
    latest_rows["raw_prob"] = model.predict_proba(X_latest)[:, 1]
    latest_rows["risk_score"] = get_risk_score(latest_rows["raw_prob"])
    latest_rows["risk_level"] = latest_rows["risk_score"].apply(get_risk_level)

    latest_rows["global_rank"] = latest_rows["risk_score"].rank(ascending=False).astype(int)

    
    def simple_driver(row):
        if row.get("debt_to_assets", 0) > 0.5:
            return "High leverage"
        elif row.get("cash_to_assets", 1) < 0.1:
            return "Low cash buffer"
        elif row.get("volatility_6m", 0) > 0.3:
            return "High volatility"
        else:
            return "Mixed signals"
            
    latest_rows["top_driver"] = latest_rows.apply(simple_driver, axis=1)

    latest_rows["debt_vs_sector"] = latest_rows["debt_to_assets"] / latest_rows.groupby("sector")["debt_to_assets"].transform("mean")
    latest_rows["cash_vs_sector"] = latest_rows["cash_to_assets"] / latest_rows.groupby("sector")["cash_to_assets"].transform("mean")

    leaderboard_cols = [
        "global_rank",      
        "ticker",
        "sector",
        "date",
        "risk_level",
        "risk_score",
        "top_driver",       
        "debt_to_assets",
        "cash_to_assets",
        "volatility_6m",
        "debt_vs_sector",   
        "cash_vs_sector"   
    ]
    leaderboard_cols = [c for c in leaderboard_cols if c in latest_rows.columns]
    # Build Top 10 Leaderboard
    leaderboard = latest_rows[leaderboard_cols].sort_values("risk_score", ascending=False).head(10)

    display = leaderboard.copy()
    
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"]).dt.strftime('%Y-%m-%d')
        
    if "risk_score" in display.columns:
        display["risk_score"] = display["risk_score"].round(1)
    
    for col in ["debt_to_assets", "cash_to_assets"]:
        if col in display.columns:
            display[col] = (pd.to_numeric(display[col], errors="coerce") * 100).round(2)
            
    for col in ["volatility_6m"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(2)
            
    # Format comparison values into intuitive multiplier form like "1.8x"
    if "debt_vs_sector" in display.columns:
        display["debt_vs_sector"] = display["debt_vs_sector"].apply(lambda x: f"{x:.1f}x" if pd.notnull(x) else "N/A")
    if "cash_vs_sector" in display.columns:
        display["cash_vs_sector"] = display["cash_vs_sector"].apply(lambda x: f"{x:.1f}x" if pd.notnull(x) else "N/A")
            
    return display, latest_rows

def build_sector_peer_comparison(df, company_data, metrics):
    sector = company_data["sector"]
    year = pd.to_datetime(company_data["date"]).year
    peers = df[(df["sector"] == sector) & (df["date"].dt.year == year)].copy()

    if peers.empty:
        peers = df[df["sector"] == sector].copy()

    rows = []
    for metric in metrics:
        company_value = safe_float(company_data.get(metric))
        sector_avg = safe_float(pd.to_numeric(peers[metric], errors="coerce").mean())
        absolute_diff = company_value - sector_avg
        if abs(sector_avg) > 1e-9:
            percent_diff = (absolute_diff / abs(sector_avg)) * 100
        else:
            percent_diff = np.nan
        rows.append({
            "metric": metric,
            "company_value": company_value,
            "sector_average": sector_avg,
            "difference": absolute_diff,
            "percent_difference": percent_diff,
        })
    return pd.DataFrame(rows), peers

def filter_display_metrics(comparison_df, peer_df, min_non_zero_share=0.05):
    keep_metrics = []
    for _, row in comparison_df.iterrows():
        metric = row["metric"] 
        peer_values = pd.to_numeric(peer_df[metric], errors="coerce")
        non_zero_share = (peer_values.fillna(0).abs() > 1e-9).mean()
        company_value = safe_float(row["company_value"])
        sector_average = safe_float(row["sector_average"])

        has_signal = abs(company_value) > 1e-9 or abs(sector_average) > 1e-9
        if has_signal and non_zero_share >= min_non_zero_share:
            keep_metrics.append(metric)

    return comparison_df[comparison_df["metric"].isin(keep_metrics)].copy()

# Initialize application
try:
    df = load_data()
    model, feature_cols = train_model(df)
except Exception as e:
    st.error(f"Failed to connect to the database or load the data: {e}")
    st.stop()

# Sidebar Setting
st.sidebar.title("Risk Screener")
st.sidebar.markdown("Filter and predict downside risk for US public companies based on 12-month forward alpha.")

available_tickers = sorted(df["ticker"].unique().tolist())
default_idx = available_tickers.index("AAPL") if "AAPL" in available_tickers else 0
selected_ticker = st.sidebar.selectbox("Searching Company Index (Ticker)", available_tickers, index=default_idx)

# Main Page: Rankings and Global Precomputation
st.title("Financial Risk Scoring Dashboard")
st.markdown("Target: Predict if a company will severely underperform the S&P 500 (`Alpha < -30%`) over the next 12 months.")
st.divider()

# Top 10 High-Risk Watchlist
st.subheader("🚨 Top 10 High-Risk Watchlist")

display_board, all_latest_scores = build_risk_leaderboard(df, model, feature_cols)

st.dataframe(display_board, use_container_width=True, hide_index=True)
st.caption("Risk score (0-100) is normalized across the entire market universe.")
st.divider()


# For SHAP based model explainability
X_pred = df[df["ticker"] == selected_ticker][feature_cols].tail(1)
X_pred = make_numeric_matrix(X_pred, feature_cols)

#  Macro to Micro: Macro View Window
st.header("🌍 Macro to Micro: Top-Down Risk Analysis")

# 1.Box Plot
st.subheader("1. Sector Risk Distribution")
st.markdown("Use boxplots to examine natural financial barriers across sectors. For example: where is the leverage ceiling for utilities?")

box_metric = st.selectbox(
    "Select metric for Sector Distribution:", 
    ["debt_to_assets", "cash_to_assets", "volatility_6m"], 
    index=0,
    format_func=lambda x: x.replace('_', ' ').title() 
)

q_high = df[box_metric].quantile(0.99)
box_df = df[df[box_metric] < q_high].copy()
clean_metric_name = box_metric.replace('_', ' ').title()
fig_box = px.box(
    box_df, 
    x="sector", 
    y=box_metric, 
    color="sector", 
    points="outliers",
    labels={
        box_metric: clean_metric_name
    }
)
fig_box.update_traces(
    hovertemplate=None, 
    hoverinfo="y+name",  
    yhoverformat=".4f"   
)

fig_box.update_layout(
    showlegend=False, 
    xaxis={'categoryorder':'median ascending', 'title': None}, 
    yaxis={'title': clean_metric_name}, 
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)", 
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
company_data = df[df["ticker"] == selected_ticker].sort_values("date", ascending=False).iloc[0]
latest_date = pd.to_datetime(company_data["date"])
current_sector = company_data["sector"]

company_risk_info = all_latest_scores[all_latest_scores["ticker"] == selected_ticker].iloc[0]
c_score = company_risk_info["risk_score"]
c_level = company_risk_info["risk_level"]


st.subheader("2. 🌌 Multi-Dimensional Risk Landscape")
st.markdown(
    "💡 Interactive Tip: \n"
    " - On the vertical axis, click and drag up or down to filter the data.\n"
    " - Dragging the top of the `Debt to Assets` axis to see which companies cluster in high-leverage ranges."
)

df_clean = df.loc[:, ~df.columns.duplicated()].copy()

par_cols = ['ticker', 'sector', 'debt_to_assets', 'cash_to_assets', 'volatility_6m']
if 'ocf_to_debt' in df_clean.columns:
    par_cols.append('ocf_to_debt')

plot_df = df_clean[par_cols].copy()
plot_df = plot_df.dropna(subset=['ticker', 'sector'])

dims_to_plot = [c for c in par_cols if c not in ['ticker', 'sector']]

for col in dims_to_plot:
    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0)


for col in dims_to_plot:
    q_low = plot_df[col].quantile(0.01)
    q_high = plot_df[col].quantile(0.95)
    plot_df[col] = plot_df[col].clip(lower=q_low, upper=q_high)

plot_df['color_val'] = 0  

sector_mask = plot_df['sector'] == current_sector
sector_df = plot_df[sector_mask].copy()

if len(sector_df) > 80:
    selected_row = sector_df[sector_df['ticker'] == selected_ticker]
    sector_df = sector_df.drop(selected_row.index, errors='ignore').sample(n=79, random_state=42)
    if not selected_row.empty:
        sector_df = pd.concat([sector_df, selected_row])
sector_df['color_val'] = 1

other_df = plot_df[~sector_mask].copy()
if len(other_df) > 150:
    other_df = other_df.sample(n=150, random_state=42)
other_df['color_val'] = 0

# Highlight the target company
sector_df.loc[sector_df['ticker'] == selected_ticker, 'color_val'] = 2
final_plot_df = pd.concat([other_df, sector_df], ignore_index=True)

fig_par = px.parallel_coordinates(
    final_plot_df,
    color="color_val",
    dimensions=dims_to_plot,
    color_continuous_scale=[(0.0, "#E0E0E0"), (0.5, "#00FFFF"), (1.0, "#FFD700")],
    labels={
        "debt_to_assets": "Debt to Assets",
        "cash_to_assets": "Cash to Assets",
        "volatility_6m": "6M Volatility",
        "ocf_to_debt": "OCF to Debt"
    }
)

fig_par.update_layout(
    coloraxis_showscale=False, 
    margin=dict(l=40, r=40, t=40, b=40),
    plot_bgcolor="rgba(0,0,0,0)", 
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_par, use_container_width=True, theme="streamlit")

st.caption(
    f"⚪ Gray: Market background ({len(other_df)} companies)"
    f"🔵 Blue: {current_sector} ({max(0, len(sector_df)-1)} companies)"
    f"🟡 Gold: Selected Company ({selected_ticker})"
)
st.divider()


# Micro-level Stocks: Key Metrics and Radar Chart

st.subheader(f"3. {selected_ticker} - Latest financial report analysis ({latest_date.date()})")

comparison_metrics = [
    "debt_to_assets",
    "cash_to_assets",
    "ocf_to_debt",
    "volatility_6m",
    "robust_lev_vol_in_teraction",
    "robust_debt_cash_interaction",
]
comparison_metrics = [m for m in comparison_metrics if m in df.columns]
comparison_df, peer_df = build_sector_peer_comparison(df, company_data, comparison_metrics)
display_metric_df = filter_display_metrics(comparison_df, peer_df)

summary = comparison_df.set_index("metric")
col1, col2, col3, col4 = st.columns(4)

debt = safe_float(summary.loc["debt_to_assets", "company_value"]) if "debt_to_assets" in summary.index else 0.0
debt_sector = safe_float(summary.loc["debt_to_assets", "sector_average"]) if "debt_to_assets" in summary.index else 0.0
cash = safe_float(summary.loc["cash_to_assets", "company_value"]) if "cash_to_assets" in summary.index else 0.0
cash_sector = safe_float(summary.loc["cash_to_assets", "sector_average"]) if "cash_to_assets" in summary.index else 0.0
vol = safe_float(summary.loc["volatility_6m", "company_value"]) if "volatility_6m" in summary.index else 0.0
vol_sector = safe_float(summary.loc["volatility_6m", "sector_average"]) if "volatility_6m" in summary.index else 0.0

col1.metric("Debt / Assets", f"{debt * 100:.1f}%", f"{(debt - debt_sector) * 100:+.2f} pp vs Sector")
col2.metric("Cash / Assets", f"{cash * 100:.1f}%", f"{(cash - cash_sector) * 100:+.2f} pp vs Sector")
col3.metric("6M Volatility", f"{vol:.2f}", f"{vol - vol_sector:+.2f} vs Sector")

# Risk Score and Level
if "High" in c_level:
    col4.error(f"{c_level} Risk\n\nScore: {c_score:.1f} / 100")
elif "Elevated" in c_level:
    col4.warning(f"{c_level} Risk\n\nScore: {c_score:.1f} / 100")
elif "Moderate" in c_level:
    col4.info(f"{c_level} Risk\n\nScore: {c_score:.1f} / 100")
else:
    col4.success(f"{c_level} Risk\n\nScore: {c_score:.1f} / 100")

st.divider()

st.subheader("Sector Peer Financial Indicator Comparison")
st.markdown(f"GICS Sector: **{current_sector}** | Peer Group Year: **{latest_date.year}**")

if len(display_metric_df) >= 3:
    metrics = display_metric_df["metric"].tolist()
    
    name_map = {}
    for m in metrics:
        clean_name = m.replace('_', ' ').title()
        if clean_name == "Robust Lev Vol Interaction":
            clean_name = "Lev Vol Interaction"
        elif clean_name == "Robust Debt Cash Interaction":
            clean_name = "Debt Cash Interaction"
        name_map[m] = clean_name
    
    display_names = [name_map[m] for m in metrics]
    
    company_ranks = []
    sector_ranks = []
    hover_texts_company = []
    hover_texts_sector = []
    
    for _, row in display_metric_df.iterrows():
        m = row['metric']
        clean_m = name_map[m] 
        
        c_val = safe_float(row['company_value'])
        s_val = safe_float(row['sector_average'])
        diff_pct = safe_float(row['percent_difference'])
        
        valid_market = pd.to_numeric(df[m], errors='coerce').dropna()
        if len(valid_market) > 0:
            c_rank = (valid_market <= c_val).mean() * 100
        else:
            c_rank = 50.0
            
        s_rank = (df[df["sector"] == current_sector][m].le(s_val).mean()) * 100
            
        company_ranks.append(c_rank)
        sector_ranks.append(s_rank)
        
        hover_c = f"theta: {clean_m}<br>r: {c_rank:.1f}% | diff: {diff_pct:+.1f}%"
        hover_s = f"theta: {clean_m} (Baseline)<br>r: {s_rank:.1f}%"
        
        hover_texts_company.append(hover_c)
        hover_texts_sector.append(hover_s)

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=sector_ranks, 
        theta=display_names, 
        fill='toself',
        fillcolor='rgba(225, 165, 0, 0.4)', 
        name=f'{current_sector} Median Baseline (50%)', 
        line_color='orange', 
        text=hover_texts_sector,
        hoverinfo='text'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=company_ranks, 
        theta=display_names, 
        fill='toself',
        fillcolor='rgba(0, 255, 255, 0.4)', 
        line_color='cyan',
        text=hover_texts_company, 
        hoverinfo='text',
        name=selected_ticker
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            showticklabels=True,
            gridcolor="rgba(128, 128, 128, 0.4)", 
        ),
        angularaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.4)", 
        ),
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=60, r=60, t=40, b=40), 
    
    font=dict(color=None), 
    legend=dict(
        font=dict(color=None),
        bgcolor="rgba(0,0,0,0)" 
    )
)
    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.sidebar.divider()
st.sidebar.info("Model powered by XGBoost & SHAP.")
st.divider()

# SHAP Local Attribution
st.subheader("SHAP Local Risk Attribution")
st.markdown("Shows which model features push this company toward higher or lower predicted downside risk.")

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    explainer = get_shap_explainer(model)
    shap_values = explainer(X_pred)

    if shap_values.feature_names is not None:
        clean_names = []
        for name in shap_values.feature_names:
            clean_name = str(name).replace('_', ' ').title()
            clean_names.append(clean_name)
        shap_values.feature_names = clean_names

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white') 
    ax.set_facecolor('white')

    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    
    fig.canvas.draw()
    
    def human_format(num):
        num = float(num)
        if abs(num) >= 1e12: return f'{num/1e12:.1f}T'
        if abs(num) >= 1e9: return f'{num/1e9:.1f}B'
        if abs(num) >= 1e6: return f'{num/1e6:.1f}M'
        return f'{num:.2f}'

    new_labels = []
    for tick in ax.get_yticklabels():
        txt = tick.get_text()
        
        txt = re.sub(r'^\d+\s+other features', 'Other Features', txt, flags=re.IGNORECASE)
        
        if "=" in txt and "Other Features" not in txt:
            parts = txt.split("=")
            try:
                val = float(parts[0].strip())
                if abs(val) > 1e6:
                    txt = f"{human_format(val)} = {parts[1].strip()}"
            except: pass
            
        new_labels.append(txt)

    ax.set_yticklabels(new_labels)
    for tick in ax.get_yticklabels():
        tick.set_color('black')
        tick.set_fontsize(11)

    for text_obj in ax.texts:
        txt = text_obj.get_text()
        if "f(x)" in txt.lower():
            text_obj.set_color('black')
            text_obj.set_fontweight('bold')

    for tick in ax.get_xticklabels():
        tick.set_color('black')

    plt.title(f"Risk Scoring Decomposition for {selected_ticker}", fontsize=12, pad=10, color='black', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP chart rendering failed: {e}")