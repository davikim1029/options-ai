# mlp_pipeline.py
import os
from pathlib import Path
import sqlite3
import pandas as pd
import requests
from shared_options.log.logger_singleton import getLogger
from utils.utils import to_native_types
import json
from constants import DB_PATH, TRAINING_DIR

logger = getLogger()

# -----------------------------
# Configuration
# -----------------------------

AI_SERVER_PORT = 8100
MIN_NEW_ROWS = 100  # number of new completed permutations before creating a file

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_new_permutations(threshold=MIN_NEW_ROWS):
    """Fetch unprocessed rows from option_permutations table"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Ensure 'processed' column exists
    c.execute("PRAGMA table_info(option_permutations)")
    existing_columns = [col[1] for col in c.fetchall()]
    if "processed" not in existing_columns:
        c.execute("ALTER TABLE option_permutations ADD COLUMN processed INTEGER DEFAULT 0")
        conn.commit()

    # Fetch unprocessed rows
    c.execute("SELECT * FROM option_permutations WHERE processed=0")
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    df = pd.DataFrame(rows, columns=columns)
    conn.close()

    if df.empty:
        logger.logMessage("No new unprocessed permutations.")
        return None

    if len(df) < threshold:
        logger.logMessage(f"Not enough new permutations: {len(df)}/{threshold}")
        return None

    return df


def mark_permutations_processed(df):
    """Mark permutations as processed in DB"""
    if df is None or df.empty:
        return
    logger.logMessage(f"Marking {len(df)} permutations as processed")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for idx in df.index:
        osi = df.at[idx, "osiKey"]
        buy_time = df.at[idx, "buy_timestamp"]
        sell_time = df.at[idx, "sell_timestamp"]
        c.execute("""
            UPDATE option_permutations
            SET processed=1
            WHERE osiKey=? AND buy_timestamp=? AND sell_timestamp=?
        """, (osi, buy_time, sell_time))
    conn.commit()
    conn.close()


def upload_to_ai_server(df: pd.DataFrame, auto_train=True):
    """Upload dataframe to AI server (MLP-ready)"""
    url = f"http://127.0.0.1:{AI_SERVER_PORT}/train/upload"
    temp_file = TRAINING_DIR / "_upload_tmp_permutations.json"

    try:
        # Convert dataframe rows to JSON lines
        with temp_file.open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                # ensure JSON-serializable types
                f.write(json.dumps(to_native_types(row)) + "\n")

        with temp_file.open("rb") as f:
            files = {"file": f}
            payload = {"auto_train": str(auto_train).lower()}
            resp = requests.post(url, files=files, data=payload)
            resp.raise_for_status()
            result = resp.json()

        temp_file.unlink(missing_ok=True)
        logger.logMessage(f"AI server upload complete. Response: {result}")
        return result

    except requests.exceptions.RequestException as e:
        logger.logMessage(f"Upload error[Client Side]: {e}")
        return {"status": "error", "message": str(e)}


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():
    # 1. Fetch unprocessed permutations
    df = fetch_new_permutations()
    if df is None:
        return

    # 2. Upload directly to AI server for training
    result = upload_to_ai_server(df)
    
    # 3. Mark permutations as processed
    mark_permutations_processed(df)

    return result
