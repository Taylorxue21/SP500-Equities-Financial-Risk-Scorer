import pandas as pd
from sqlalchemy import create_engine
import requests
import io
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")  
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print("Downloading the latest S&P 500 GICS industry classification data from Wikipedia...")

# 1. Wikipedia URL for S&P 500 companies
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# 2. Fetch the HTML content and read the first table (which contains the S&P 500 companies)
response = requests.get(url, headers=headers)
tables = pd.read_html(io.StringIO(response.text))
sp500_table = tables[0]

# 3. Clean and prepare the DataFrame
df_profiles = sp500_table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].copy()
df_profiles = df_profiles.rename(columns={
    'Symbol': 'ticker',
    'Security': 'company_name',
    'GICS Sector': 'sector',
    'GICS Sub-Industry': 'industry'
})

# 4. Write to PostgreSQL database
df_profiles.to_sql('company_profiles', engine, if_exists='replace', index=False)

print(f"🎉 Success! A total of {len(df_profiles)} companies' industry data has been written to the company_profiles table.")