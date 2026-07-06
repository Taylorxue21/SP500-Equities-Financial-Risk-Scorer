import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import os

from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


df = pd.read_parquet("data/processed/model_dataset.parquet")

df['date'] = pd.to_datetime(df['date'])

print("\n====== CURRENT COLUMNS ======")
print(df.columns.tolist())
print("=============================\n")

macro_cols = [c for c in ["interest_rate", "inflation_yoy"] if c in df.columns]
if macro_cols:
    print(f"✅ Macro features found in model dataset: {macro_cols}")
else:
    print("⚠️ No macro features found in model dataset. Run fetch_macro.py and build_dataset.py if needed.")

# Feature Engineering
df = df.sort_values(['ticker', 'date'])

df['debt_change'] = df.groupby('ticker')['debt_to_assets'].diff()
df['cash_change'] = df.groupby('ticker')['cash_to_assets'].diff()
df['income_change'] = df.groupby('ticker')['net_income'].diff()
df['vol_change'] = df.groupby('ticker')['volatility_6m'].diff()

df['debt_trend_3'] = (
    df.groupby('ticker')['debt_to_assets']
    .rolling(3)
    .mean()
    .reset_index(0, drop=True)
)

df = df.fillna(0)

# Time Split (Critical Fix)
train_df = df[df['date'] <= '2020-12-31']
valid_df = df[(df['date'] > '2020-12-31') & (df['date'] <= '2022-12-31')]
test_df  = df[df['date'] >= '2023-01-01']

print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
print(
    "Date ranges | "
    f"Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} | "
    f"Valid: {valid_df['date'].min().date()} -> {valid_df['date'].max().date()} | "
    f"Test: {test_df['date'].min().date()} -> {test_df['date'].max().date()}"
)

# Features / Label
drop_cols = [
    'ticker', 'date',
    'stock_12m_return', 'spy_12m_return', 'alpha',
    'risk_label'
]

X_train = pd.get_dummies(train_df.drop(columns=drop_cols, errors='ignore'))
X_valid = pd.get_dummies(valid_df.drop(columns=drop_cols, errors='ignore'))
X_test  = pd.get_dummies(test_df.drop(columns=drop_cols, errors='ignore'))

# Align Columns
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)
X_train, X_test  = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = train_df['risk_label']
y_valid = valid_df['risk_label']
y_test  = test_df['risk_label']

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / max(positive_count, 1)
print(f"Train risk labels | positive={positive_count}, negative={negative_count}, scale_pos_weight={scale_pos_weight:.2f}")


# Model Training 
model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)


# Evaluation
def evaluate(name, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob > 0.5).astype(int)

    print(f"\n===== {name} =====")
    print("PR-AUC:", round(average_precision_score(y, prob), 4))
    print("ROC-AUC:", round(roc_auc_score(y, prob), 4))
    print("Top 10% precision:", round(top_bucket_precision(y, prob, 0.10), 4))
    print("Top 20% precision:", round(top_bucket_precision(y, prob, 0.20), 4))
    print(classification_report(y, pred, zero_division=0))

    return prob

def top_bucket_precision(y_true, prob, bucket_size):
    scored = pd.DataFrame({"y": y_true.reset_index(drop=True), "risk_prob": prob})
    n_top = max(1, int(len(scored) * bucket_size))
    return scored.sort_values("risk_prob", ascending=False).head(n_top)["y"].mean()

valid_probs = evaluate("VALID", X_valid, y_valid)
test_probs  = evaluate("TEST", X_test, y_test)


# Risk Leaderboard
test_df = test_df.copy()
test_df["risk_prob"] = test_probs

leaderboard = (
    test_df[[
        "ticker",
        "sector",
        "risk_prob",
        "debt_to_assets",
        "volatility_6m",
        "cash_to_assets"
    ]]
    .sort_values("risk_prob", ascending=False)
    .head(20)
)

print("\n🚨 TOP 20 HIGH RISK COMPANIES")
print(leaderboard)

os.makedirs("reports", exist_ok=True)
leaderboard.to_csv("reports/risk_leaderboard.csv", index=False)


# Save Model Output
joblib.dump(model, "xgboost_risk_model.pkl")
test_df.to_parquet("data/processed/test_scored.parquet")

print("\n✅ Done: model + leaderboard saved")

# Shap Explainability 

explainer = shap.TreeExplainer(model)

X_sample = X_test.sample(min(500, len(X_test)), random_state=42)

shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("reports/shap_summary.png", bbox_inches="tight")
plt.close()


# Average Impact 
shap_mean = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    "feature": X_sample.columns,
    "shap_importance": shap_mean
}).sort_values("shap_importance", ascending=False)

shap_importance.to_csv("reports/shap_feature_importance.csv", index=False)

print("✅ SHAP analysis saved")