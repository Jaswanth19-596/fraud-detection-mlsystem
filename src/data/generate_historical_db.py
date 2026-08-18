import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import yaml
from sqlalchemy import Column, String, Float, Integer, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Base = declarative_base()

class TransactionRecord(Base):
    __tablename__ = 'transactions'
    
    transaction_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, index=True)
    amount = Column(Float)
    true_label = Column(Integer, nullable=True) # 1 for fraud, 0 for normal
    ml_prediction = Column(Integer, nullable=True) # For phase 2
    
    # V1-V28 Features
    v1 = Column(Float); v2 = Column(Float); v3 = Column(Float); v4 = Column(Float)
    v5 = Column(Float); v6 = Column(Float); v7 = Column(Float); v8 = Column(Float)
    v9 = Column(Float); v10 = Column(Float); v11 = Column(Float); v12 = Column(Float)
    v13 = Column(Float); v14 = Column(Float); v15 = Column(Float); v16 = Column(Float)
    v17 = Column(Float); v18 = Column(Float); v19 = Column(Float); v20 = Column(Float)
    v21 = Column(Float); v22 = Column(Float); v23 = Column(Float); v24 = Column(Float)
    v25 = Column(Float); v26 = Column(Float); v27 = Column(Float); v28 = Column(Float)

def load_config(config_path: str = "configs/params.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_transaction(timestamp: datetime, fraud_ratio: float) -> TransactionRecord:
    is_fraud = np.random.rand() < fraud_ratio
    
    mean_offset = 2.0 if is_fraud else 0.0
    v_features = np.random.normal(loc=mean_offset, scale=1.0, size=28).tolist()
    amount = round(np.random.lognormal(mean=3.0, sigma=1.0), 2)
    
    return TransactionRecord(
        transaction_id=str(uuid.uuid4()),
        timestamp=timestamp,
        amount=amount,
        true_label=1 if is_fraud else 0,
        v1=v_features[0], v2=v_features[1], v3=v_features[2], v4=v_features[3],
        v5=v_features[4], v6=v_features[5], v7=v_features[6], v8=v_features[7],
        v9=v_features[8], v10=v_features[9], v11=v_features[10], v12=v_features[11],
        v13=v_features[12], v14=v_features[13], v15=v_features[14], v16=v_features[15],
        v17=v_features[16], v18=v_features[17], v19=v_features[18], v20=v_features[19],
        v21=v_features[20], v22=v_features[21], v23=v_features[22], v24=v_features[23],
        v25=v_features[24], v26=v_features[25], v27=v_features[26], v28=v_features[27]
    )

def main():
    config = load_config()
    db_conn = config['infrastructure']['db_conn']
    fraud_ratio = config['simulation']['fraud_ratio']
    
    # We will generate 90 days of data
    days_to_generate = 90
    total_records = 250000  # Generate 250k rows for realistic ML training

    logging.info(f"Connecting to DB to generate static historical data: {db_conn}")
    engine = create_engine(db_conn)
    Base.metadata.drop_all(engine) # Start fresh
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    start_time = datetime.now() - timedelta(days=days_to_generate)
    end_time = datetime.now()
    
    logging.info(f"Generating {total_records} historical records between {start_time.date()} and {end_time.date()}...")
    
    batch = []
    batch_size = 5000
    
    # Time delta between records
    time_step = (end_time - start_time) / total_records
    current_time = start_time
    
    for i in range(total_records):
        record = generate_transaction(current_time, fraud_ratio)
        batch.append(record)
        current_time += time_step
        
        if len(batch) >= batch_size:
            session.bulk_save_objects(batch)
            session.commit()
            logging.info(f"Inserted {i+1}/{total_records} records...")
            batch = []
            
    if batch:
        session.bulk_save_objects(batch)
        session.commit()
        
    logging.info("Static historical data generation complete!")
    session.close()

if __name__ == "__main__":
    main()
