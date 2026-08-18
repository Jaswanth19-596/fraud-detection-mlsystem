# Knowledge Graph

## Entities
- **Historical DB Generator** (`src/data/generate_historical_db.py`): One-off script to simulate pre-existing static transactions with confirmed labels.
- **Relational DB** (`postgres` in `docker-compose.yml`): Stores transactions. Holds confirmed historical data and future live predictions.
- **Data Extractor** (`src/data/extract_training_data.py`): Pulls mature historical data into Parquet.
- **Model** (Pending - Will be PyTorch / Scikit-learn)
- **Live Inference API** (Pending - Will consume from Kafka and predict)
- **Message Queue** (`kafka` in `docker-compose.yml`): Streaming broker for live transactions (Pending integration).

## Edges
- **Historical DB Generator** -> [inserts static data] -> **Relational DB**
- **Data Extractor** -> [SQL query] -> **Relational DB**
- **Data Extractor** -> [creates snapshot] -> `data/raw/train_snapshot.parquet`
- **train_snapshot.parquet** -> [trains] -> **Model**

## Data Flows
- **Phase 1 (Initial Training)**: `generate_historical_db.py` -> PostgreSQL -> `extract_training_data.py` -> Parquet File (DVC tracked) -> Model Training.
- **Phase 2 (Live Inference - Pending)**: Live Transaction -> Kafka -> FastAPI -> Model Prediction -> PostgreSQL (unlabeled).
- **Phase 3 (Delayed Labels - Pending)**: Analyst / Customer -> PostgreSQL (updates true_label).
- **Phase 4 (Retraining)**: PostgreSQL -> `extract_training_data.py` -> DVC Snapshot -> Model Retraining.
