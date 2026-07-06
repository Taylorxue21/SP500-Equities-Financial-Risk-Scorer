<div align="right">
  🌐 <b>Language:</b>
  <a href="README.md">English</a> | <a href="README_ko.md">한국어</a> | 简体中文
</div>

# 📈 S&P500股票金融风险评分系统

<div align="right">
  🌐 <b>语言:</b>
  <a href="README.md">English</a> | 한국어 | 简体中文
</div>

<p align="left">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/Model-XGBoost-orange.svg" alt="XGBoost"></a>
  <a href="https://shap.readthedocs.io/"><img src="https://img.shields.io/badge/XAI-SHAP-red.svg" alt="SHAP"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

> 一个端到端的数据分析项目：从财务报表与市场数据出发，识别未来 12 个月可能大幅跑输大盘的美股上市公司，并通过交互式看板将结果转化为可执行的业务洞察。


## 1. 业务背景与问题定义

**业务问题：**
能否仅利用公开的财务报表、市场行为数据和宏观经济指标，提前识别未来一年可能出现严重跑输大盘风险的美股上市公司？

**目标变量定义：**

```
risk_label = 1  如果  未来12个月相对收益(相对SPY) <= -30%
risk_label = 0  否则
```

这不是一个“炫技型”机器学习项目，而是一次**面向业务决策的数据分析实践**：核心价值不在模型本身，而在于——

- 如何把原始财务数据转化为**可解释、可比较**的分析指标（同比增长率、行业相对分位数、杠杆/现金结构等）；
- 如何用**统计与可视化方法**验证特征是否真的和风险相关，而不是拍脑袋建模；
- 如何把模型输出**翻译成业务语言**（风险分数、行业对比、驱动因子），让非技术背景的人也能看懂并使用。

---

## 2. 分析框架（Analytics Workflow）

```
数据采集层        →  财务报表 / 历史行情 / 宏观指标（FMP API + FRED API）
   ↓
数据清洗与建模层  →  PostgreSQL 存储 + SQL 特征工程（避免未来函数 / 严格按报告日对齐）
   ↓
探索性分析层      →  行业分布箱线图 / 时间序列标签率 / 相关性与分位数分析
   ↓
建模与验证层      →  逻辑回归基线 vs XGBoost 主模型，时间序列切分，PR-AUC / 精确率 / Top-K 命中率
   ↓
可解释性层        →  SHAP 全局特征重要性 + 单公司局部归因（waterfall）
   ↓
业务呈现层        →  Streamlit 交互看板：风险排行榜 / 行业对比 / 雷达图 / 归因解释
```

---

## 3. 数据说明

| 维度 | 说明 |
|---|---|
| 覆盖范围 | 美股上市公司（优先 S&P 500 子集，100–200 只流动性较好的标的） |
| 观测频率 | 季度财务报表 + 日度行情聚合 |
| 时间跨度 | 2015–2024 |
| 基准指数 | SPY（标普 500 ETF） |
| 数据来源 | Financial Modeling Prep（财务与行情）、FRED（宏观经济指标） |

### 特征体系（4 大类，20+ 指标）

- **盈利与成长**：毛利率、营业利润率、净利率、ROE、ROA、营收同比增速
- **偿债与流动性**：流动比率、现金比率、`debt_to_assets`、`cash_to_assets`
- **现金流质量**：自由现金流利润率、经营现金流/负债（`ocf_to_debt`）、负FCF标记
- **市场行为**：1M/3M/6M/12M 收益率、波动率、最大回撤、相对 SPY 超额收益
- **宏观环境**：联邦基金利率、通胀率、失业率、10Y-2Y 利差
- **行业相对化特征**：同行业内的杠杆分位数、ROE 分位数等（消除行业本身的系统性差异）

> ⚠️ 数据处理的核心原则：**严格防止未来函数（look-ahead bias）**——所有特征只使用 `as_of_date` 当天及之前可获得的信息，财报数据按保守滞后对齐。这是金融类分析项目中最容易被面试官追问、也最容易暴露基本功的细节。

---

## 4. 建模与评估

采用**时间序列切分**而非随机切分，更贴近真实业务场景（用历史预测未来，而非"用未来数据泄漏式验证"）：

```
训练集：2015–2020
验证集：2021–2022
测试集：2023–2024
```

| 模型 | 用途 |
|---|---|
| Logistic Regression | 基线模型，验证特征本身的线性可分性，便于业务解释系数方向 |
| XGBoost（`scale_pos_weight` 处理类别不平衡） | 主模型，捕捉非线性交互（如"高杠杆 × 高波动"的组合风险） |

**评估指标（金融风控场景更看重排序能力而非单点准确率）：**

- PR-AUC（正样本稀少，比 ROC-AUC 更有参考价值）
- Top 10% / Top 20% 高风险分桶精确率 —— **业务解读：在模型判定为最高风险的前 10% 公司中，有多少比例真的在未来 12 个月内跑输大盘超过 30%？**
- Recall / Precision / F1 / 混淆矩阵

> 📊 *实际结果（请在完成一次完整跑批后填入真实数值，不要用占位数字投简历）：*
> - PR-AUC：`__`
> - Top 10% 分桶精确率：`__`
> - XGBoost 相对逻辑回归基线提升：`__`

---

## 5. 可解释性分析（SHAP）

模型不是黑箱交付，而是配套完整的归因分析：

- **全局层面**：SHAP summary plot / bar plot，识别对整体风险预测贡献最大的特征（例如高杠杆、低现金缓冲、高波动率的组合信号）
- **个股层面**：针对看板中选中的任意公司，生成 SHAP waterfall 图，逐项展示哪些财务/市场特征把这家公司推向了"高风险"或"低风险"

这一层是整个项目**从"模型"走向"决策工具"**的关键——风控或投资人员不需要理解 XGBoost 原理，只需要看懂"这家公司为什么被标记为高风险"。

---

## 6. 交互式分析看板（Streamlit Dashboard）

看板围绕"**宏观 → 中观（行业）→ 微观（个股）**"的分析逻辑设计，这也是数据分析师日常向业务方汇报时最常用的叙事结构：

1. **Top 10 高风险公司观察榜**：全市场风险分数（0–100 相对分位）排序 + 主要风险驱动标签
2. **行业风险分布（箱线图）**：观察不同 GICS 行业在杠杆、现金、波动率上的"天然边界"（例如公用事业行业的杠杆中枢天然更高）
3. **多维风险全景图（平行坐标图）**：交互式筛选，观察高杠杆公司在多个维度上的聚集特征
4. **个股 vs 行业同行对比**：所选公司的核心指标与所属行业均值的差异（百分点 / 倍数表达），并叠加雷达图展示全市场分位排名
5. **SHAP 个股风险归因**：可视化解释单一公司的风险分数是如何由具体财务指标"拼装"出来的

**技术实现要点：**
- PostgreSQL 作为数据层，`SQLAlchemy` 连接，附带本地 CSV 兜底加载（保障线上 Demo 稳定性，避免数据库连接失败导致页面白屏）
- `Plotly` 实现交互式箱线图 / 平行坐标图 / 雷达图，`Matplotlib` 承载 SHAP 静态图
- 缓存策略：`st.cache_data` / `st.cache_resource` 分离数据加载与模型训练，避免每次交互重复计算

---

## 7. 技术栈

| 类别 | 工具 |
|---|---|
| 数据采集 | Financial Modeling Prep API、FRED API |
| 数据存储与建模 | PostgreSQL、SQL（特征工程全部下沉到 SQL 层，而非全部堆在 Python） |
| 数据处理 | Python（pandas、numpy） |
| 建模 | scikit-learn（Logistic Regression）、XGBoost |
| 可解释性 | SHAP |
| 可视化与看板 | Streamlit、Plotly、Matplotlib |
| 工程化 | dotenv 环境变量管理、模块化脚本（`fetch_*` / `clean_*` / `build_*` / `train_*`） |

---

## 8. 项目结构

```
us-public-company-financial-risk-scoring/
├── config/
│   ├── tickers_sp500.csv
│   └── fred_series.csv
├── sql/
│   ├── 01_create_tables.sql
│   ├── 03_build_financial_features.sql
│   ├── 04_build_market_features.sql
│   ├── 05_build_macro_features.sql
│   └── 06_build_model_dataset.sql
├── src/
│   ├── fetch_fmp.py
│   ├── fetch_fred.py
│   ├── clean_financials.py
│   ├── clean_prices.py
│   ├── build_features.py
│   ├── build_labels.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── notebooks/
│   ├── 01_data_quality_check.ipynb
│   ├── 03_modeling_baseline.ipynb
│   └── 04_lightgbm_shap.ipynb
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── business_memo_en.md
│   ├── business_memo_kr.md
│   └── model_card.md
├── data/
│   ├── raw/
│   └── processed/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 9. 如何运行

```bash
# 1. 克隆项目并安装依赖
git clone <repo-url>
cd us-public-company-financial-risk-scoring
pip install -r requirements.txt

# 2. 配置环境变量（API Key、数据库连接）
cp .env.example .env

# 3. 数据采集
python src/fetch_fmp.py
python src/fetch_fred.py

# 4. 建表与特征构建
psql -f sql/01_create_tables.sql
python src/build_features.py
python src/build_labels.py

# 5. 训练与评估
python src/train.py
python src/evaluate.py

# 6. 启动看板
streamlit run app/streamlit_app.py
```

---

## 10. 局限性与未来规划

**当前局限（诚实呈现，面试时主动说出来反而加分）：**
- 样本量集中在流动性较好的大盘股，对小盘股 / 次新股的泛化能力未验证
- 财务数据存在报告滞后与口径调整风险，跨公司比较未做行业会计准则差异修正
- 未纳入文本类信息（如财报电话会情绪、新闻舆情），风险信号目前仅来自结构化数据

**下一步计划：**
- [ ] 补充 LightGBM 与 XGBoost 的对比实验，量化两者在 PR-AUC / 推理速度上的差异
- [ ] 引入行业中性化（sector-neutral）标签定义，剔除行业 Beta 对风险标签的干扰
- [ ] 接入财报文本 / 新闻情绪特征，验证是否提升 Top 10% 分桶精确率
- [ ] 增加模型稳定性监控（PSI / 特征漂移检测），为看板加入"模型健康度"模块

---

## 12. 免责声明

本项目仅用于学习与作品集展示，不构成任何投资建议。模型基于历史数据训练，不保证未来表现，请勿用于实际投资决策。
