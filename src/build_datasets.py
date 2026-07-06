import os
import dotenv
import pandas as pd 
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")  
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def generate_model_dataset():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    sql_path = os.path.join("sql","06_build_model_dataset.sql")
    print(f"Executing SQL commands from {sql_path}...")
    with open(sql_path, 'r', encoding='utf-8') as f:
        with engine.begin() as conn:
            conn.execute(text(f.read()))

    print("Extracting the full wide table from PostgreSQL...")
    query = "SELECT * FROM model_dataset ORDER BY ticker, date;"
    df_model = pd.read_sql(query, engine)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    out_path = os.path.join("data", "processed", "model_dataset.parquet")
    df_model.to_parquet(out_path, index=False)

    print("Success! The model training dataset has been saved to: {out_path}")
    print(f"Dataset size: {df_model.shape[0]} rows x {df_model.shape[1]} columns")
    print(f"High-risk samples (risk_label=1): {df_model['risk_label'].sum()}")

if __name__ == "__main__":
    generate_model_dataset()
