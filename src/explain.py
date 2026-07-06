import shap
import pandas as pd
import xgboost as xgb

df = pd.read_parquet("data/processed/model_dataset.parquet")

y = df["risk_label"]
X = df.drop(columns=["risk_label", "ticker", "date"], errors="ignore")

model = xgb.XGBClassifier()
model.load_model("model.json")  

model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 1. Global interpretation 
shap.summary_plot(shap_values, X, show=False)
import matplotlib.pyplot as plt
plt.savefig("reports/shap_summary.png")

# 2. Single-company interpretation
i = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[i],
    X.iloc[i],
    matplotlib=True
)
plt.savefig("reports/shap_single_company.png")