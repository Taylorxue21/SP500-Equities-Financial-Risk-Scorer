import os 
import pandas as pd
from fredapi import Fred
from sqlalchemy import create_engine
from dotenv import load_dotenv

def get_db_engine():
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")  
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def fetch_macro_data():
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("❌ FRED_API_KEY is not set in the environment variables.")
        return
    
    fred_client = Fred(api_key=api_key)

    # Notice how everything below is now indented inside the function
    try:
        # 1. Fetch Federal Funds Rate (monthly)
        print("Fetching Federal Funds Rate data...")
        fed_funds = fred_client.get_series('FEDFUNDS', observation_start='2010-01-01')

        # 2. Fetch CPI inflation data (monthly)
        print("Fetching CPI Inflation data...")
        cpi = fred_client.get_series('CPIAUCSL', observation_start='2010-01-01')
        # Calculate CPI YoY Inflation Rate
        inflation_yoy = cpi.pct_change(periods=12) * 100

        # 3. Merge DataFrames
        macro_df = pd.DataFrame({
            'interest_rate': fed_funds,
            'inflation_yoy': inflation_yoy
        }).dropna()

        macro_df.index.name = 'observation_date'
        macro_df = macro_df.reset_index()
        macro_df['observation_date'] = pd.to_datetime(macro_df['observation_date'])

        print('\n Data preview:')
        print(macro_df.tail())

        # 4. Write to PostgreSQL
        engine = get_db_engine()
        macro_df.to_sql('macro_data', engine, if_exists='replace', index=False)
        print("Macro data successfully written to PostgreSQL table 'macro_data'.")
    except Exception as e:
        print(f"❌ An error occurred while fetching or writing macro data: {e}")

if __name__ == "__main__":
    fetch_macro_data()