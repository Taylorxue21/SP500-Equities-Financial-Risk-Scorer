CREATE INDEX IF NOT EXISTS idx_raw_stock_prices_ticker ON raw_stock_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_stock_prices_date ON raw_stock_prices(date);
CREATE INDEX IF NOT EXISTS idx_raw_stock_prices_ticker_date ON raw_stock_prices(ticker, date);

DROP TABLE IF EXISTS derived_risk_labels CASCADE;

CREATE TABLE derived_risk_labels AS
WITH base_window AS(
    SELECT
        f.ticker,
        f.date AS statement_date,
        --Set the target end date to 12 months after the reference date.
        (f.date + INTERVAL '12 months')::DATE AS target_end_date,

        -- Find the stock price on the earnings release date or the first trading day thereafter (P0).
        (SELECT "adjClose" FROM raw_stock_prices p
         WHERE p.ticker = f.ticker AND p.date >= f.date 
           AND p.date <= f.date + INTERVAL '14 days' 
         ORDER BY p.date ASC LIMIT 1) AS stock_p0,

        (SELECT "adjClose" FROM raw_stock_prices p
         WHERE p.ticker = f.ticker AND p.date >= (f.date + INTERVAL '12 months')::DATE 
           AND p.date <= (f.date + INTERVAL '12 months') + INTERVAL '14 days'
         ORDER BY p.date ASC LIMIT 1) AS stock_p12,

         -- Find the market benchmark (SPY) price on the same earnings release date.
        (SELECT "adjClose" FROM raw_stock_prices p
         WHERE p.ticker = 'SPY' AND p.date >= f.date 
           AND p.date <= f.date + INTERVAL '14 days'
         ORDER BY p.date ASC LIMIT 1) AS spy_p0,

         -- Find the market benchmark (SPY) price 12 months after the earnings release date, or on the first trading day thereafter (SPY12).
        (SELECT "adjClose" FROM raw_stock_prices p
         WHERE p.ticker = 'SPY' AND p.date >= (f.date + INTERVAL '12 months')::DATE 
           AND p.date <= (f.date + INTERVAL '12 months') + INTERVAL '14 days'
         ORDER BY p.date ASC LIMIT 1) AS spy_p12
    FROM derived_financial_features f
),
returns_calc AS(
    SELECT
        ticker,
        statement_date,
        -- Calculate the stock return over the 12-month period.
        (stock_p12 - stock_p0) / NULLIF(stock_p0, 0) AS stock_12m_return,
        -- Calculate the market benchmark (SPY) return over the same 12-month period.
        (spy_p12 - spy_p0) / NULLIF(spy_p0, 0) AS spy_12m_return
    FROM base_window
    --Filter out the most recent earnings records that do not yet have future price data available.
    WHERE stock_p12 IS NOT NULL AND spy_p12 IS NOT NULL
)
SELECT
    ticker,
    statement_date,
    stock_12m_return,
    spy_12m_return,
    --Calculate excess return (Alpha) = stock return - market benchmark return.
    (stock_12m_return - spy_12m_return) AS alpha,
    
    -- Generate the binary target label for machine learning.
    -- Label as high risk (1) if the stock underperforms the market benchmark by more than 5% over the next 12 months; otherwise, label as low risk (0).
    CASE 
        WHEN (stock_12m_return - spy_12m_return) <= -0.30 THEN 1 
        ELSE 0 
    END AS risk_label
FROM returns_calc;

CREATE INDEX idx_derived_risk_labels_ticker ON derived_risk_labels(ticker);
CREATE INDEX idx_derived_risk_labels_statement_date ON derived_risk_labels(statement_date);
