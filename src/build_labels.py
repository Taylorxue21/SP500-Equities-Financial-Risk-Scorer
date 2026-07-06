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

def run_labeling():
    engine = get_db_engine()
    sql_path = os.path.join("sql", "04_build_risk_labels.sql")

    print(f"📂 Executing SQL commands from {sql_path}...")
    with open(sql_path, 'r', encoding='utf-8') as f:
        sql_commands = f.read()

    with engine.begin() as conn:
        conn.execute(text(sql_commands))
    print(f"✅ Successfully executed SQL commands from {sql_path}.")

def verify_labels():
    engine = get_db_engine()
    print("Verifying the labels table in PostgreSQL...")
    
    query = """
    SELECT ticker, statement_date, stock_12m_return AS stock_ret, 
           spy_12m_return AS spy_ret, alpha, risk_label
    FROM derived_risk_labels
    WHERE alpha IS NOT NULL AND alpha != 0
    ORDER BY ticker, statement_date
    LIMIT 10;
    """

    df = pd.read_sql(query, engine)
    print(df.to_string(index=False))
    print("\n--------------------------------------------------------------")

if __name__ == "__main__":
    try:
        run_labeling()
        verify_labels()
    except Exception as e:
        print(f"❌ An error occurred: {e}")

