DROP TABLE IF EXISTS model_dataset CASCADE;

CREATE TABLE model_dataset AS
-- 1. Based on the latest as_of_date (disclosure date), precisely locate trading day prices for T0 and T12
WITH price_base AS (
    SELECT 
        f.*,
        -- Find stock price on disclosure date shift to the next trading day if market is closed on that day
        (SELECT "adjClose" FROM raw_stock_prices p 
         WHERE p.ticker = f.ticker AND p.date >= f.date 
         ORDER BY p.date ASC LIMIT 1) AS price_t0,
         
        -- Find stock price 1 year after disclosure date
        (SELECT "adjClose" FROM raw_stock_prices p 
         WHERE p.ticker = f.ticker AND p.date >= (f.date + INTERVAL '12 months')::DATE 
         ORDER BY p.date ASC LIMIT 1) AS price_t12,

        -- Retrieve SPY market index price at T0
        (SELECT "adjClose" FROM raw_stock_prices p 
         WHERE p.ticker = 'SPY' AND p.date >= f.date 
         ORDER BY p.date ASC LIMIT 1) AS spy_t0,

        -- Retrieve SPY market index price at T12 
        (SELECT "adjClose" FROM raw_stock_prices p 
         WHERE p.ticker = 'SPY' AND p.date >= (f.date + INTERVAL '12 months')::DATE 
         ORDER BY p.date ASC LIMIT 1) AS spy_t12

    FROM derived_financial_features f
),

-- 2. Calculate holding period return
return_calc AS (
    SELECT 
        *,
        (price_t12 - price_t0) / NULLIF(price_t0, 0) AS stock_12m_return,
        (spy_t12 - spy_t0) / NULLIF(spy_t0, 0) AS spy_12m_return
    FROM price_base
)

-- 3. Output final risk control wide table. Automatically filter out rows with missing prices due to overly old historical data
SELECT 
    r.ticker,
    r.date,
    r.sector,
    r.total_revenue,
    r.net_income,
    r.cash_to_assets,
    r.debt_to_assets,
    r.ocf_to_debt,
    r.volatility_6m,
    r.robust_lev_vol_interaction,
    r.robust_debt_cash_interaction,
    m.interest_rate,
    m.inflation_yoy,
    r.stock_12m_return,
    r.spy_12m_return,
    (r.stock_12m_return - r.spy_12m_return) AS alpha,
    -- Core fix: Define labels based on Alpha. Completely remove market exposure influence
    CASE WHEN (r.stock_12m_return - r.spy_12m_return) < -0.30 THEN 1 ELSE 0 END AS risk_label
FROM return_calc r
LEFT JOIN LATERAL (
    SELECT
        interest_rate,
        inflation_yoy
    FROM macro_data m
    WHERE m.observation_date <= r.date
    ORDER BY m.observation_date DESC
    LIMIT 1
) m ON TRUE
WHERE r.stock_12m_return IS NOT NULL AND r.spy_12m_return IS NOT NULL;
