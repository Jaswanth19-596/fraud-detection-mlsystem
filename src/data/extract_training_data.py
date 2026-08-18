import logging
import os
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import yaml
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str = "configs/params.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    db_conn = config['infrastructure']['db_conn']
    
    maturity_days = config['data_ingestion']['maturity_delay_days']
    window_days = config['data_ingestion']['retraining_window_days']
    
    output_dir = config['data_ingestion']['output_dir']
    output_filename = config['data_ingestion']['output_filename']
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Connecting to DB to extract training data: {db_conn}")
    engine = create_engine(db_conn)
    
    # Calculate cutoff dates
    now = datetime.now()
    end_date = now - timedelta(days=maturity_days)
    start_date = end_date - timedelta(days=window_days)
    
    query = f"""
        SELECT 
            transaction_id, timestamp, amount, 
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28,
            true_label as class
        FROM transactions
        WHERE timestamp >= '{start_date.isoformat()}'
          AND timestamp <= '{end_date.isoformat()}'
    """
    
    logging.info(f"Executing query for transactions between {start_date.date()} and {end_date.date()}...")
    
    df = pd.read_sql(query, engine)
    
    logging.info(f"Extracted {len(df)} transactions.")
    
    if len(df) > 0:
        logging.info(f"Saving to {output_path} (Parquet format)...")
        df.to_parquet(output_path, engine='pyarrow', index=False)
        logging.info("Snapshot complete!")
    else:
        logging.warning("No data found in the specified window. Please run the producer with backfill.")

if __name__ == "__main__":
    main()
