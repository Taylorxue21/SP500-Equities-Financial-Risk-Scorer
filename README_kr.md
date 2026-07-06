<div align="right">
  🌐 <b>Language:</b>
  <a href="README.md">English</a> | 한국어 | <a href="README_zh.md">简体中文</a>
</div>

# 📈 S&P 500 주식 재무 리스크 평가 시스템

<p align="left">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/Model-XGBoost-orange.svg" alt="XGBoost"></a>
  <a href="https://shap.readthedocs.io/"><img src="https://img.shields.io/badge/XAI-SHAP-red.svg" alt="SHAP"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

> 본 프로젝트는 미국 상장 기업을 대상으로 하는 XGBoost 기반의 설명 가능한 재무 리스크 분석 시스템입니다. 다차원 시장 데이터, 기본 재무제표 및 거시 경제 지표를 통합하여 바이사이드(Buy-side) 수준의 Streamlit 대화형 분석 대시보드를 구축합니다.

> **목표**: 특정 기업이 향후 12개월 내에 S&P 500 대비 크게 하회할 것인지(`Alpha < -30%`) 예측하고, SHAP 국소적 기여도를 통해 그 원인을 설명합니다.

## 1. 비즈니스 배경 및 문제 정의

**비즈니스 문제:**
공개된 재무제표, 시장 행동 데이터 및 거시 경제 지표만을 활용하여, 향후 1년 동안 시장 평균을 크게 하회할 위험이 있는 미국 상장 기업을 사전에 식별할 수 있는가?

**목표 변수 정의:**

(`risk_label = 1`) 향후 12개월간 상대 수익률(SPY 대비) <= -30% 인 경우, 그렇지 않으면 (`risk_label = 0`)  


본 학습 프로젝트는 **비즈니스 의사결정을 지향하는 데이터 분석 실무**입니다. 핵심 가치는 모델 자체에 있는 것이 아니라 다음 요소들에 있습니다:

- 원시 재무 데이터를 **해석 및 비교 가능한** 분석 지표(예: 전년 동기 대비 성장률, 산업 내 상대 분위수, 레버리지/현금 구조)로 변환하는 방법;
- **통계 및 시각화 방법**을 사용하여 특징(feature)이 실제로 리스크와 상관관계가 있는지 검증하는 방법;
- 모델의 산출물을 **비즈니스 언어로 번역**하는 방법 (예: 리스크 점수, 산업 비교, 주요 동인)

---

## 2. 분석 프레임워크 (Analytics Workflow)

```
데이터 수집 계층                →  재무제표 / 과거 시세 / 거시 지표 (FMP API + FRED API)
↓
데이터 정제 및 모델링 계층      →  PostgreSQL 저장소 + SQL 특징 공학 (미래 참조 편향 방지 / 보고일 기준 엄격한 정렬)
↓
탐색적 분석 계층                →  산업별 분포 박스 플롯 / 시계열 레이블 비율 / 상관관계 및 분위수 분석
↓
모델링 및 검증 계층             →  로지스틱 회귀 베이스라인 vs XGBoost 메인 모델, 시계열 분할, PR-AUC / 정밀도 / Top-K 적중률
↓
해석 가능성 계층                →  SHAP 전역 특징 중요도 + 단일 기업 국소적 기여도 (워터폴 차트)
↓
비즈니스 프레젠테이션 계층      →  Streamlit 대화형 대시보드: 리스크 리더보드 / 산업 비교 / 레이더 차트 / 기여도 설명
```

---

## 3. 데이터 설명

| 차원 | 설명 |
|---|---|
| 커버리지 | 미국 상장 기업 (S&P 500 하위 집합 우선, 유동성이 높은 100–200개 종목) |
| 관측 주기 | 분기별 재무제표 + 일별 시세 집계 |
| 시간 범위 | 2015–2025 |
| 벤치마크 지수 | SPY (S&P 500 ETF) |
| 데이터 출처 | Financial Modeling Prep (재무 및 시세 데이터), FRED (거시 경제 지표) |

### 특징 체계 (4대 범주, 20+ 지표)

- **수익성 및 성장성**: 매출총이익률, 영업이익률, 순이익률, ROE, ROA, 전년 동기 대비(YoY) 매출 성장률
- **지불 능력 및 유동성**: 유동비율, 당좌비율(현금비율), `debt_to_assets`, `cash_to_assets`
- **현금흐름 품질**: 잉여현금흐름(FCF) 마진, 영업현금흐름/부채(`ocf_to_debt`), 마이너스 FCF 플래그
- **시장 행동**: 1M/3M/6M/12M 수익률, 변동성, 최대 낙폭(MDD), SPY 대비 상대 초과 수익률
- **거시 경제 환경**: 연방기금 금리, 인플레이션율, 실업률, 10년-2년 국채 장단기 금리차
- **산업 상대 특징**: 산업 내 레버리지 분위수, ROE 분위수 등 (산업 고유의 체계적 차이 제거)

> **미래 참조 편향(Look-ahead bias) 방지** — 모든 특징은 `as_of_date` 당일 및 그 이전에 활용 가능한 정보만 엄격하게 사용하며, 재무제표 데이터는 보수적인 지연(lagging)을 적용하여 정렬합니다.

---

## 4. 모델링 및 평가 (Modeling & Evaluation)

실제 비즈니스 시나리오에 더 부합하도록 무작위 분할 대신 시계열 분할(time series splitting)을 채택했습니다 ('미래 데이터를 사용한 데이터 누수 검증' 대신 '과거 데이터를 사용하여 미래 예측'):

학습 데이터셋 (Training Set): 2015–2020
검증 데이터셋 (Validation Set): 2021–2022
테스트 데이터셋 (Test Set): 2023–2025


| 모델 | 용도 |
|---|---|
| 로지스틱 회귀 (Logistic Regression) | 베이스라인 모델. 특징(feature)의 선형 분리 가능성을 검증하고, 계수 방향에 대한 비즈니스적 해석을 용이하게 합니다. |
| XGBoost (클래스 불균형 처리를 위한 `scale_pos_weight` 적용) | 메인 모델. 비선형 상호작용(예: "높은 레버리지 × 높은 변동성"의 결합 리스크)을 포착합니다. |

### 꼬리 위험(Tail Risk) 의사결정 임계값 최적화

전통적인 분류기는 일반적으로 `0.5`의 확률 임계값을 기본으로 사용하지만, 꼬리 위험(tail risk) 관리에서 이 설정은 상업적으로 용납될 수 없습니다. 아래의 정밀도-재현율 곡선 (Precision-Recall Curve)에서 볼 수 있듯이 자산의 하방 리스크는 유의미한 비선형적 특성을 나타냅니다. 검증 데이터셋에서 F1-Score를 최대화함으로써 시스템은 최종적으로 최적의 리스크 트리거 임계값을 **`0.0839`**로 식별했습니다.

| 정밀도-재현율 곡선 (AUC = 0.959) | 최적화된 '풍동 실험' 혼동 행렬 |
| :---: | :---: |
| <img src="docs/images/pr_curve.png" width="500" /> | <img src="docs/images/confusion_matrix_optimal.png" width="500" /> |

* **하방 보호 능력 (재현율 = 97.03%)**: `0.0839`의 시스템 수준 임계값에서 모델은 블라인드 테스트 데이터셋에 있는 절대 낙폭 샘플(Alpha ≤ -30%)의 97%를 성공적으로 포착하고 차단했습니다.

* **신호 신뢰성 (정밀도 = 85.45%)**: 공격적인 리스크 방어 전략을 유지하면서도 신호는 여전히 85.4%의 정밀도를 유지하여, 오탐지(false alarms)를 크게 줄이고 포트폴리오의 기회비용을 효과적으로 통제했습니다.

---

## 5. 해석 가능성 분석 (SHAP)

모델은 블랙박스 형태로 제공되지 않으며, 포괄적인 기여도 분석과 함께 제공됩니다:

- **전역 수준 (Global Level)**: SHAP 요약 플롯(summary plot) / 바 플롯(bar plot). 전체 리스크 예측에 가장 크게 기여하는 특징을 식별합니다 (예: 높은 레버리지, 낮은 현금 버퍼, 높은 변동성의 결합 신호).
- **개별 종목 수준 (Stock Level)**: 대시보드에서 선택한 모든 기업에 대해 SHAP 워터폴 플롯(waterfall plot)을 생성하여, 어떤 재무/시장 특징이 해당 기업을 "고위험" 또는 "저위험"으로 밀어넣었는지 항목별로 보여줍니다.

이 계층은 전체 프로젝트가 **"모델"에서 "의사결정 도구"로** 나아가는 핵심입니다. 리스크 관리나 투자 담당자는 XGBoost의 원리를 이해할 필요 없이, "왜 이 기업이 고위험으로 분류되었는지"만 이해하면 됩니다.

<p align="center">
  <img src="docs/images/shap_waterfall.png" width="85%" alt="SHAP Local Waterfall Explanation">
  <br>
  <i style="color: gray; font-size: 14px;">그림: SHAP 개별 종목 리스크 기여도 - 최종 리스크 점수에 대한 다양한 재무 및 거시 지표의 구체적인 기여도를 분해하여 표시</i>
</p>


## 6. 대화형 분석 대시보드 (Streamlit Dashboard)

이 대시보드는 "**거시(Macro) → 중시(산업, Meso) → 미시(개별 종목, Micro)**"의 분석 논리를 중심으로 설계되었습니다:

1. **상위 10개 고위험 기업 관심 목록 (Watchlist)**: 전체 시장 리스크 점수(0–100 상대 분위수) 순위 + 주요 리스크 유발 레이블
2. **산업별 리스크 분포 (박스 플롯)**: 레버리지, 현금 및 변동성 측면에서 다양한 GICS 산업의 "자연적 경계" 관찰 (예: 유틸리티 산업의 레버리지 중심은 자연적으로 더 높음)
3. **다차원 리스크 파노라마 (평행 좌표 플롯)**: 다차원에 걸친 고레버리지 기업의 군집 특징을 관찰하기 위한 대화형 필터링
4. **개별 종목 vs 동종 산업 피어(Peers) 비교**: 선택한 기업의 핵심 지표와 산업 평균 간의 차이(퍼센트 포인트 / 배수로 표현)를 표시하고, 전체 시장 분위수 순위를 보여주는 레이더 차트 오버레이
5. **SHAP 개별 종목 리스크 기여도**: 단일 기업의 리스크 점수가 특정 재무 지표에 의해 어떻게 "조립"되었는지 시각적으로 설명

**주요 기술적 구현 요점:**
- 데이터 계층으로 PostgreSQL을 사용하고 `SQLAlchemy`를 통해 연결하며, 로컬 CSV 대체 로딩(fallback loading) 기능 제공 (온라인 데모의 안정성을 보장하고 데이터베이스 연결 실패로 인한 빈 화면 방지)
- `Plotly`를 사용하여 대화형 박스 플롯 / 평행 좌표 플롯 / 레이더 차트 구현; `Matplotlib`을 통해 SHAP 정적 플롯 제공
- 캐싱 전략: `st.cache_data` / `st.cache_resource`를 활용하여 데이터 로딩과 모델 학습을 분리함으로써 매 상호작용마다 반복적인 계산 방지

<p align="center">
  <img src="docs/images/radar_chart.png" width="45%" alt="Company vs Sector Radar Chart">
  <img src="docs/images/parallel_coordinates.png" width="45%" alt="Multi-dimensional Risk Parallel Coordinates">
  <br>
  <i style="color: gray; font-size: 14px;">그림: 개별 종목 vs 동종 산업 피어 비교 (왼쪽) 및 다차원 리스크 파노라마 (오른쪽)</i>
</p>



## 7. 기술 스택 (Tech Stack)

| 카테고리 | 도구 (Tools) |
|---|---|
| 데이터 수집 | Financial Modeling Prep API, FRED API |
| 데이터 저장 및 모델링 | PostgreSQL, SQL (특징 공학(Feature engineering)을 Python에 의존하지 않고 모두 SQL 계층으로 내림) |
| 데이터 처리 | Python (pandas, numpy) |
| 모델링 | scikit-learn (Logistic Regression), XGBoost |
| 해석 가능성 | SHAP |
| 시각화 및 대시보드 | Streamlit, Plotly, Matplotlib |
| 엔지니어링 | 환경 변수 관리를 위한 dotenv, 모듈화된 스크립트 (`fetch_*` / `clean_*` / `build_*` / `train_*`) |

---

## 8. 프로젝트 구조
```text
us-public-company-financial-risk-scoring/
├── .vscode/
│   └── settings.json
├── app/
│   └── streamlit_sector_peer_app.py          
├── data/
│   ├── processed/                             # 특징 저장소 (모델에서 직접 사용할 수 있는 데이터셋)
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
│   └── 03_modeling_xgboost.ipynb              # 학습 완료된 XGBoost 모델 (운영 환경 배포용)
├── reports/                                   
│   ├── feature_importance.png
│   ├── risk_leaderboard.csv
│   ├── risk_report.py
│   └── top_features.csv
├── sql/                                        # 데이터베이스 특징 공학 스크립트
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
├── .env.example                              
├── dataset.csv                              
├── requirements.txt                           
├── README.md                                   
├── README_zh.md                                
└── README_kr.md
```

## 9. 실행 방법 (How to Run)

```bash
# 1. 저장소를 클론(clone)하고 종속성을 설치합니다
git clone <repo-url>
cd us-public-company-financial-risk-scoring
pip install -r requirements.txt

# 2. 환경 변수 구성 (API 키, 로컬 경로 등)
cp .env.example .env
# 참고: 로컬에서 .env 파일을 편집하여 API 키를 구성하세요

# 3. 엔드투엔드(End-to-end) 데이터 수집 파이프라인 실행 (API)
python src/fetch_sp500_sectors.py
python src/fetch_sec_ultimate.py
python src/fetch_prices.py
python src/fetch_macro.py

# 4. 특징 및 레이블 행렬 구축 (로컬 저장소 또는 PostgreSQL 푸시다운)
# SQL을 사용하여 구축하는 경우, 데이터베이스에서 sql/ 디렉토리의 스크립트를 순서대로 실행하세요
python src/build_features.py
python src/build_labels.py
python src/build_dataset.py

# 5. 모델 학습, 평가 및 SHAP 기여도 내보내기
python src/train.py
python src/explain.py

# 6. Streamlit 대화형 대시보드 실행
streamlit run app/streamlit_sector_peer_app.py
```
---

10. 한계점 및 향후 계획 (Limitations and Future Plans)

**현재의 한계점:**

- 샘플 크기가 유동성이 높은 대형주에 집중되어 있어, 소형주 / 신규 상장주에 대한 일반화 성능은 검증되지 않았습니다.
- 재무 데이터는 보고 지연 및 지표 조정의 영향을 받습니다. 기업 간 비교 시 산업별 회계 기준의 차이는 보정되지 않았습니다.
- 텍스트 정보(예: 실적 발표 통화 센티먼트, 뉴스 센티먼트)는 포함되어 있지 않으며, 리스크 신호는 현재 구조화된 데이터에만 의존합니다.

---
## ⚠️ 면책 조항

본 프로젝트는 오직 교육 및 연구 목적으로만 제공됩니다.

시스템에서 생성된 리스크 점수, 특징 기여도(SHAP) 및 모든 분석 결과는 데이터 기반의 재무 모델링 방법을 시연하기 위한 용도로만 사용되며, 어떠한 투자, 거래 또는 재무적 조언으로 간주되지 않습니다.

금융 시장은 불확실성이 매우 높습니다. 과거 데이터를 기반으로 구축된 모델은 본질적인 한계를 가지고 있으며, 미래의 시장 성과에 대한 정확한 예측을 보장할 수 없습니다.

사용자는 실제 투자 결정을 내릴 때 본 시스템에 의존해서는 안 되며, 재무 관련 결정을 내리기 전에 반드시 전문 재무 상담사와 상담해야 합니다.

---

# 👨‍💻 작성자

**설염문(Xue Yanwen)** 

고려대학교 통계학과 
