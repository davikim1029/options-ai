# pipeline.py
import sqlite3
import pandas as pd
from pathlib import Path
import requests
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
DB_PATH = Path("../option_data_project/database/options.db")  # adjust path
TRAINING_DIR = Path("training")
AI_SERVER_PORT = 8100
MIN_NEW_OPTIONS = 20  # number of new completed options before creating a file

TRAINING_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_new_lifespans(threshold=MIN_NEW_OPTIONS):
    """Fetch unprocessed completed options from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Make sure we have a 'processed' column
    c.execute("ALTER TABLE IF NOT EXISTS option_lifespans ADD COLUMN processed INTEGER DEFAULT 0")
    conn.commit()
    
    # Select unprocessed options
    c.execute("SELECT * FROM option_lifespans WHERE processed=0")
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    
    if len(rows) < threshold:
        print(f"Not enough new options: {len(rows)}/{threshold}")
        return None
    return pd.DataFrame(rows, columns=columns)

def transform_for_fusion(df):
    """
    Transform lifespan data into the format expected by the fusion AI model.
    Each option becomes a single row containing:
      - static features: optionType, strikePrice, moneyness
      - sequence: snapshot series flattened (lastPrice, delta, gamma, etc.)
      - targets: recommendation, expectedHoldDays
    """
    processed_rows = []
    for _, row in df.iterrows():
        # Here, we simulate sequence by splitting totalSnapshots into equal parts
        sequence_data = []
        for i in range(int(row["totalSnapshots"])):
            sequence_data.append({
                "lastPrice": row.get("startPrice") + i*(row.get("endPrice") - row.get("startPrice"))/row.get("totalSnapshots"),
                "delta": row.get("avgDelta"),
                "gamma": row.get("avgGamma"),
                "theta": row.get("avgTheta"),
                "vega": row.get("avgVega"),
                "rho": row.get("avgRho"),
                "iv": row.get("avgIV"),
                "bid": row.get("startPrice"),
                "ask": row.get("endPrice"),
                "bidSize": 0,
                "askSize": 0,
                "volume": row.get("avgVolume"),
                "openInterest": row.get("avgOpenInterest"),
                "midPrice": row.get("avgMidPrice"),
                "moneyness": row.get("avgMoneyness"),
                "daysToExpiration": i
            })
        processed_rows.append({
            **{
                "osiKey": row["osiKey"],
                "strikePrice": row["strikePrice"],
                "optionType": row["optionType"],
                "moneyness": row.get("avgMoneyness"),
                "recommendation": row.get("totalChange"),  # e.g., total profit
                "expectedHoldDays": (row.get("endDate") - row.get("startDate")).days if isinstance(row.get("endDate"), pd.Timestamp) else row.get("totalSnapshots")
            },
            "sequence": sequence_data
        })
    return processed_rows

def save_csv_for_training(data):
    """Flatten and save as CSV for AI upload."""
    rows = []
    for entry in data:
        flat_row = {
            "osiKey": entry["osiKey"],
            "strikePrice": entry["strikePrice"],
            "optionType": entry["optionType"],
            "moneyness": entry["moneyness"],
            "recommendation": entry["recommendation"],
            "expectedHoldDays": entry["expectedHoldDays"]
        }
        # Flatten sequence data
        for idx, step in enumerate(entry["sequence"]):
            for k, v in step.items():
                flat_row[f"{k}_{idx}"] = v
        rows.append(flat_row)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = TRAINING_DIR / f"lifespan_training_{timestamp}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    print(f"Training CSV saved to {file_path}")
    return file_path

def upload_to_ai_server(csv_path, auto_train=True):
    url = f"http://127.0.0.1:{AI_SERVER_PORT}/train/upload"
    with open(csv_path,"rb") as f:
        files = {"file": f}
        data = {"auto_train": str(auto_train).lower()}
        try:
            resp = requests.post(url, files=files, data=data)
            print(resp.json())
        except requests.exceptions.RequestException as e:
            print(f"Error uploading training file: {e}")

def mark_processed(df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for osi in df["osiKey"]:
        c.execute("UPDATE option_lifespans SET processed=1 WHERE osiKey=?", (osi,))
    conn.commit()
    conn.close()

# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():
    df = fetch_new_lifespans()
    if df is None:
        return
    processed_data = transform_for_fusion(df)
    csv_path = save_csv_for_training(processed_data)
    upload_to_ai_server(csv_path)
    mark_processed(df)

if __name__=="__main__":
    run_pipeline()
