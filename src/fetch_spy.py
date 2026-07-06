import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")  
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Your Tiingo API KEY
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print("Downloading SPY benchmark data via Tiingo API...")

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Token {TIINGO_API_KEY}'
}

url = "https://api.tiingo.com/tiingo/daily/SPY/prices?startDate=2015-01-01&endDate=2026-06-30"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    df_spy = pd.DataFrame(data)
    
    # Format data
    df_spy_clean = pd.DataFrame({
        'date': pd.to_datetime(df_spy['date']).dt.date,
        'ticker': 'SPY',
        'close': df_spy['close'],
        'adjClose': df_spy['adjClose']
    })

    print(f"✅ Successfully downloaded {len(df_spy_clean)} days of SPY data. Inserting into database...")
    df_spy_clean.to_sql('raw_stock_prices', engine, if_exists='append', index=False)
else:
    print(f"❌ Failed to downloading, status code: {response.status_code}")
    print(response.text)