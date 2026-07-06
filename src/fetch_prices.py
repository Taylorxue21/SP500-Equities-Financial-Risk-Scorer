import os
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def get_db_engine():
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")  
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    return create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def get_all_tickers_from_db(engine):
    """Extract the full list of entities that need to be downloaded"""
    query = "SELECT DISTINCT ticker FROM derived_financial_features ORDER BY ticker;"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result]

def get_downloaded_tickers(engine):
    """Extract the list of tickers that have already been successfully downloaded (core for resuming interrupted downloads)"""
    try:
        query = "SELECT DISTINCT ticker FROM raw_stock_prices;"
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return [row[0] for row in result]
    except Exception:
        return []

def fetch_all_stock_prices_from_tiingo(start_date="2021-01-01"):
    load_dotenv()
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        print("❌ TIINGO_API_KEY is not set in your .env file!")
        return

    engine = get_db_engine()
    
    print("🔍 Checking database for existing records...")
    all_tickers = get_all_tickers_from_db(engine)
    downloaded_tickers = get_downloaded_tickers(engine)
    
    # Set difference: keep only stocks that have NOT been downloaded yet
    missing_tickers = [t for t in all_tickers if t not in downloaded_tickers]
    
    print(f"Total Tickers: {len(all_tickers)}")
    print(f"Already Downloaded: {len(downloaded_tickers)}")
    print(f"Remaining to Fetch: {len(missing_tickers)}")

    if not missing_tickers:
        print("All tickers have already been downloaded! Nothing to do.")
        return

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {api_key}'
    }

    all_prices = []
    
    for i, ticker in enumerate(missing_tickers, 1):
        print(f"⏳ [{i}/{len(missing_tickers)}] Requesting {ticker}...")
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}"
        
        try:
            while True:
                response = requests.get(url, headers=headers)
                
                # Tiingo API rate limit constraint, 50 requests per hour
                if response.status_code == 429:
                    if all_prices:
                        temp_df = pd.concat(all_prices, ignore_index=True)
                        temp_df.to_sql('raw_stock_prices', engine, if_exists='append', index=False)
                        all_prices = [] 
                        print("Intermediate data has been stored in PostgreSQL")
                    
                    print("Force sleep for 61 minutes to wait for API quota reset...")
                    time.sleep(3660) 
                    continue 
                break
                
            if response.status_code == 200:
                data = response.json()
                if not data:
                    print(f"⚠️ No price data for {ticker}")
                    continue
                
                df = pd.DataFrame(data)
                df['ticker'] = ticker
                df = df[['date', 'ticker', 'close', 'adjClose']]
                df['date'] = pd.to_datetime(df['date']).dt.date
                all_prices.append(df)
            else:
                print(f"❌ Failed for {ticker}: Status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            
        time.sleep(1.2) 

    if all_prices:
        final_df = pd.concat(all_prices, ignore_index=True)
        print(f"\n Saving final {len(final_df)} new rows to PostgreSQL...")
        final_df.to_sql('raw_stock_prices', engine, if_exists='append', index=False)
        print("✅ All stock prices successfully fetched and saved!")
    else:
        print("✅ Batch run completed.")

if __name__ == "__main__":
    fetch_all_stock_prices_from_tiingo()
