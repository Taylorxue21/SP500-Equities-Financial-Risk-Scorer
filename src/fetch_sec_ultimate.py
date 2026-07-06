import os 
import time
import requests
import pandas as pd
from functools import reduce
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")  
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}")

HEADERS = {
    'User-Agent': 'Your Name, name,  your.email@example.com',
    'Accept-Encoding': 'gzip, deflate'
}

def get_target_tickers():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT ticker FROM company_profiles", conn)
    return df['ticker'].str.replace('.', '-').tolist()

def get_sec_cik_mapping():
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    return {value['ticker']: str(value['cik_str']).zfill(10) for key, value in data.items()}

def extract_and_align_metric(facts, metric_candidates, col_name):
    """Extract clean USD numeric values and attach fiscal year (fy) + filing date for time-series alignment"""
    records = []
    for candidate in metric_candidates:
        if candidate in facts:
            if 'USD' in facts[candidate]['units']:
                for item in facts[candidate]['units']['USD']:
                    if item.get('form') == '10-K' and item.get('fp') == 'FY' and 'fy' in item:
                        records.append({
                            'fy': item['fy'],
                            'date': item['end'],
                            'filed': item.get('filed', '1900-01-01'),
                            col_name: item['val']
                        })
    
    if not records:
        return pd.DataFrame(columns=['fy', 'date', 'filed', col_name])

    df = pd.DataFrame(records)
    # Sort by filing date and keep only the latest report within each fiscal year
    df = df.sort_values(by='filed').drop_duplicates(subset=['fy'], keep='last')
    return df[['fy', 'date', 'filed', col_name]]

def fetch_and_load_sec_data(tickers):
    cik_mapping = get_sec_cik_mapping()
    all_records = []
    
    metrics_dict = {
        'total_revenue': ['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
        'gross_profit': ['GrossProfit'],
        'operating_income': ['OperatingIncomeLoss'],
        'net_income': ['NetIncomeLoss', 'ProfitLoss'],
        'total_assets': ['Assets'],
        'current_assets': ['AssetsCurrent'],
        'current_liabilities': ['LiabilitiesCurrent'],
        'operating_cashflow': ['NetCashProvidedByOperatingActivities'],
        'short_term_debt': ['DebtCurrent', 'ShortTermBorrowings', 'CommercialPaper'],
        'long_term_debt': ['LongTermDebt', 'LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations'],
        'cash_and_equivalents': ['CashAndCashEquivalentsAtCarryingValue', 'CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations']
    }

    print(f"Prepare to extract core financial features for all companies from SEC database. Total number of tickers: {len(tickers)}...")

    for i, ticker in enumerate(tickers):
        if ticker not in cik_mapping:
            continue

        cik = cik_mapping[ticker]
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 404: continue
            response.raise_for_status()

            facts = response.json().get('facts', {}).get('us-gaap', {})
            if not facts: continue

            dfs = []
            for col_name, candidates in metrics_dict.items():
                df_temp = extract_and_align_metric(facts, candidates, col_name)
                if not df_temp.empty:
                    dfs.append(df_temp)

            if not dfs:
                continue

            anchor_dates = dfs[0][['fy', 'date', 'filed']].drop_duplicates('fy')
            
            dfs_for_merge = []
            for df in dfs:
                dfs_for_merge.append(df.drop(columns=['date', 'filed']))

            df_merged = reduce(lambda left, right: pd.merge(left, right, on=['fy'], how='outer'), dfs_for_merge)
            df_merged = pd.merge(df_merged, anchor_dates, on='fy', how='left')

            for col in metrics_dict.keys():
                if col not in df_merged.columns:
                    df_merged[col] = 0.0

            # Time axis correction. Filter invalid filing dates and convert to as_of_date
            df_merged = df_merged[df_merged['filed'] != '1900-01-01'].copy()
            df_merged.rename(columns={'filed': 'as_of_date', 'date': 'fiscal_end_date'}, inplace=True)
            
            df_merged['total_debt'] = df_merged['short_term_debt'].fillna(0) + df_merged['long_term_debt'].fillna(0)
            
            df_merged = df_merged.drop(columns=['short_term_debt', 'long_term_debt'], errors='ignore')
            
            df_merged['ticker'] = ticker
            all_records.append(df_merged)

            time.sleep(0.15) 

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")

    if all_records:
        final_df = pd.concat(all_records, ignore_index=True)
        final_df['as_of_date'] = pd.to_datetime(final_df['as_of_date']).dt.date
        final_df['fiscal_end_date'] = pd.to_datetime(final_df['fiscal_end_date']).dt.date
        
        for col in metrics_dict.keys():
            if col not in final_df.columns and col not in ['short_term_debt', 'long_term_debt']:
                final_df[col] = 0.0

        print(f"\n Finalized dataset is ready writing {len(final_df)} authoritative records into PostgreSQL...")

        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS raw_financials CASCADE;"))
        
        final_df.to_sql('raw_financials', engine, if_exists='append', index=False)
        print(" SEC full dataset extraction successful! ")
    else:
        print("❌ No valid SEC data was fetched for the provided tickers.")

if __name__ == '__main__':
    sp500_tickers = get_target_tickers()
    fetch_and_load_sec_data(sp500_tickers)