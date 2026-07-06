import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")  
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def get_db_engine():
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def execute_sql_file(filename):
    """Read and execute SQL commands from a file."""
    engine = get_db_engine()
    sql_path = os.path.join("sql", filename)

    print(f"Executing SQL commands from {sql_path}...")
    with open(sql_path, 'r', encoding='utf-8') as f:
        sql_commands = f.read()
    
    with engine.begin() as conn:
        conn.execute(text(sql_commands))
    print(f"✅ Successfully executed SQL commands from {sql_path}.")

    
def verify_features():
    """Verify that the features table exists and contains data."""
    engine = get_db_engine()
    print("Verifying the features table in PostgreSQL...")
    
    query = """
        SELECT ticker, date,
               ROUND(cash_to_assets::NUMERIC, 4) AS cash_ratio,
               ROUND(debt_to_assets::NUMERIC, 4) AS leverage,
               ROUND(robust_lev_vol_interaction::NUMERIC, 4) AS risk_factor
        FROM derived_financial_features
        WHERE debt_to_assets IS NOT NULL
        LIMIT 5;
        """
    df = pd.read_sql(query, engine)
    print(df.to_string(index=False))
    print("\n-----------")

if __name__ == "__main__":
    try:
        execute_sql_file("03_build_financial_features.sql")
        verify_features()
    except Exception as e:
        print(f"❌ An error occurred: {e}")