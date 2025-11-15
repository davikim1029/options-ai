# pipeline.py
# Current file directory
import os
from pathlib import Path
here = os.path.dirname(os.path.abspath(__file__))
# Go up one level (from options-alert to options) and then into option-file-server
db_path = os.path.join(here, "..", "option-file-server", "database", "options.db")

DB_PATH = os.path.normpath(db_path)
TRAINING_DIR = Path("training")
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from shared_options.log.logger_singleton import getLogger
from ai_model_service import transform_for_fusion_streaming
from utils.utils import write_sequence_streaming

logger = getLogger()

# -----------------------------
# Configuration
# -----------------------------

AI_SERVER_PORT = 8100
MIN_NEW_OPTIONS = 20  # number of new completed options before creating a file

TRAINING_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_new_lifetimes(threshold=MIN_NEW_OPTIONS):
    """Fetch unprocessed completed options from the database, enriched with totalSnapshots."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Make sure we have a 'processed' column
    c.execute("PRAGMA table_info(option_lifetimes)")
    columns_info = c.fetchall()
    existing_columns = [col[1] for col in columns_info]  # col[1] is the column name

    if "processed" not in existing_columns:
        c.execute("ALTER TABLE option_lifetimes ADD COLUMN processed INTEGER DEFAULT 0")
        conn.commit()

    # Select unprocessed options
    c.execute("SELECT * FROM option_lifetimes WHERE processed=0")
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    df = pd.DataFrame(rows, columns=columns)

    # If no unprocessed data, bail early
    if df.empty:
        logger.logMessage("No new unprocessed lifetimes.")
        conn.close()
        return None

    # --- Enrich with totalSnapshots ---
    snapshot_counts = pd.read_sql_query(
        "SELECT osiKey, COUNT(*) as totalSnapshots FROM option_lifetimes GROUP BY osiKey",
        conn
    )
    df = df.merge(snapshot_counts, on="osiKey", how="left")
    df["totalSnapshots"] = df["totalSnapshots"].fillna(0).astype(int)

    conn.close()

    # Check threshold
    if len(df) < threshold:
        logger.logMessage(f"Not enough new options: {len(df)}/{threshold}")
        return None

    return df


def mark_processed(df):
    logger.logMessage("Marking keys as processed")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for osi in df["osiKey"]:
        c.execute("UPDATE option_lifetimes SET processed=1 WHERE osiKey=?", (osi,))
    conn.commit()
    conn.close()
    logger.logMessage("Marked keys as processed")
    
import json
import requests

def upload_to_ai_server_csv_sequence(generator, auto_train=True):
    """Upload raw option sequences to AI server; server will flatten + compute targets"""
    url = f"http://127.0.0.1:{AI_SERVER_PORT}/train/upload"
    logger.logMessage("Uploading sequence data to AI server...")

    try:
        temp_file = TRAINING_DIR / f"_upload_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Write JSON streaming directly from generator
        write_sequence_streaming(temp_file, generator, logger=logger)

        with open(temp_file, "rb") as f:
            files = {"file": f}
            payload = {"auto_train": str(auto_train).lower()}
            resp = requests.post(url, files=files, data=payload)
        
        resp.raise_for_status()
        result = resp.json()

        temp_file.unlink(missing_ok=True)
        return result

    except requests.exceptions.RequestException as e:
        logger.logMessage(f"Upload error: {e}")
        return {"status": "error", "message": str(e)}


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():
    # 1. Fetch unprocessed completed options from DB
    df = fetch_new_lifetimes()
    if df is None:
        return

    # 2. Transform into sequence-based structure
    stream = transform_for_fusion_streaming(df, logger=logger)

    # 3. Upload directly to AI server (CSV flattened internally in server now)
    result = upload_to_ai_server_csv_sequence(stream)
    logger.logMessage(f"Training server responded: {result}")

    # 4. Mark options as processed in DB
    mark_processed(df)
