# pipeline.py
import sqlite3
import time
import pandas as pd
from pathlib import Path
import requests
from datetime import datetime
import os
from shared_options.log.logger_singleton import getLogger
import tempfile

logger = getLogger()

# -----------------------------
# Configuration
# -----------------------------
# Current file directory
here = os.path.dirname(os.path.abspath(__file__))

# Go up one level (from options-alert to options) and then into option-file-server
db_path = os.path.join(here, "..", "option-file-server", "database", "options.db")

DB_PATH = os.path.normpath(db_path)
TRAINING_DIR = Path("training")
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

def transform_for_fusion(df):
    """
    Transform the lifetime snapshot table into the format expected by the fusion AI model.

    Each option (osiKey) becomes a single row containing:
      - static features: strikePrice, optionType, moneyness
      - sequence: ordered snapshots for all relevant features
      - targets: recommendation, expectedHoldDays
    """
    processed_rows = []
    skipped = 0

    # Group all snapshots by osiKey
    grouped = df.groupby("osiKey", sort=True)
    total_groups = len(grouped)
    logger.logMessage(f"{total_groups} options to process")
    osi_cnt = 0
    for osiKey, group in grouped:
        group = group.sort_values("timestamp")
        osi_cnt += 1
        
        # Ensure chronological order
        if len(group) == 0:
            skipped += 1
            continue
        # Sequence data for all snapshots
        logger.logMessage(f"Processing option  {osiKey} | {osi_cnt}/{total_groups}")
        sequence_data = []
        for _, snap in group.iterrows():
            sequence_data.append({
                "lastPrice": snap["lastPrice"],
                "bid": snap["bid"],
                "ask": snap["ask"],
                "bidSize": snap["bidSize"],
                "askSize": snap["askSize"],
                "volume": snap["volume"],
                "openInterest": snap["openInterest"],
                "nearPrice": snap["nearPrice"],
                "inTheMoney": snap["inTheMoney"],
                "delta": snap["delta"],
                "gamma": snap["gamma"],
                "theta": snap["theta"],
                "vega": snap["vega"],
                "rho": snap["rho"],
                "iv": snap["iv"],
                "daysToExpiration": snap["daysToExpiration"],
                "spread": snap["spread"],
                "midPrice": snap["midPrice"],
                "moneyness": snap["moneyness"]
            })

        # Targets: we can compute recommendation as total profit/change across the lifetime
        first_price = group.iloc[0]["lastPrice"]
        last_price = group.iloc[-1]["lastPrice"]
        recommendation = last_price - first_price  # simple placeholder; replace with your formula

        expected_hold_days = len(group)

        # Static features: take first snapshot (or compute averages if preferred)
        first_snap = group.iloc[0]

        processed_rows.append({
            "osiKey": osiKey,
            "strikePrice": first_snap["strikePrice"],
            "optionType": first_snap["optionType"],
            "moneyness": first_snap["moneyness"],
            "recommendation": recommendation,
            "expectedHoldDays": expected_hold_days,
            "sequence": sequence_data
        })

    logger.logMessage(f"Processed {len(processed_rows)} options, skipped {skipped} with no snapshots")
    return processed_rows


def save_csv_for_training(data, logger, TRAINING_DIR: Path):
    """
    Flatten and save lifetime sequences as CSV for AI upload,
    streaming incrementally (no huge list in memory).
    Each chunk is flushed safely to disk, so the system never hangs.
    """
    logger.logMessage("Saving CSV")

    cnt = 0
    total = len(data)
    throttle_every = 1000   # throttle every N items
    delay = 0.05            # seconds to sleep per throttle
    chunk_size = 5000       # write every N rows
    skipped = 0

    # Prepare directories and file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRAINING_DIR.mkdir(exist_ok=True)
    final_path = TRAINING_DIR / f"lifetime_training_{timestamp}.csv"

    # Create a temporary file to ensure atomic writes
    with tempfile.NamedTemporaryFile("w", delete=False, dir=TRAINING_DIR, newline="") as tmp_file:
        writer = None
        batch = []

        for entry in data:
            cnt += 1
            try:
                flat_row = {
                    "osiKey": entry["osiKey"],
                    "strikePrice": entry["strikePrice"],
                    "optionType": entry["optionType"],
                    "moneyness": entry["moneyness"],
                    "recommendation": entry.get("recommendation", 0),
                    "expectedHoldDays": entry.get("expectedHoldDays", len(entry["sequence"]))
                }

                # Flatten sequence
                for idx, step in enumerate(entry["sequence"]):
                    for k, v in step.items():
                        flat_row[f"{k}_{idx}"] = v

                batch.append(flat_row)

            except Exception as e:
                skipped += 1
                logger.logMessage(f"⚠️ Skipped entry {cnt} due to error: {e}")
                continue

            # Write chunk to disk
            if len(batch) >= chunk_size:
                df = pd.DataFrame(batch)
                if writer is None:
                    df.to_csv(tmp_file, index=False)
                    writer = True
                else:
                    df.to_csv(tmp_file, index=False, header=False)
                tmp_file.flush()
                batch.clear()

            # Logging & throttling
            if cnt % 100 == 0:
                logger.logMessage(f"Processed {cnt}/{total} entries ({skipped} skipped)...")
            if cnt % throttle_every == 0:
                time.sleep(delay)

        # Write remaining rows
        if batch:
            df = pd.DataFrame(batch)
            if writer is None:
                df.to_csv(tmp_file, index=False)
            else:
                df.to_csv(tmp_file, index=False, header=False)
            tmp_file.flush()

    # Atomically move temp file into final position
    Path(tmp_file.name).replace(final_path)
    logger.logMessage(f"✅ Training CSV saved to {final_path} ({cnt - skipped} rows, {skipped} skipped)")

    return final_path.name


def upload_to_ai_server(csv_path, auto_train=True):
    """Upload CSV to AI server for training."""
    url = f"http://127.0.0.1:{AI_SERVER_PORT}/train/upload"
    logger.logMessage("Uploading...")
    try:
        full_path = Path("training") / csv_path
        with open(full_path, "rb") as f:
            files = {"file": f}
            data = {"auto_train": str(auto_train).lower()}
            resp = requests.post(url, files=files, data=data)
        resp.raise_for_status()
        result = resp.json()
        logger.logMessage(f"Upload result: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.logMessage(f"Error uploading training file: {e}")
        return {"status": "error", "message": str(e)}

def mark_processed(df):
    logger.logMessage("Marking keys as processed")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for osi in df["osiKey"]:
        c.execute("UPDATE option_lifetimes SET processed=1 WHERE osiKey=?", (osi,))
    conn.commit()
    conn.close()
    logger.logMessage("Marked keys as processed")

# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():
    df = fetch_new_lifetimes()
    if df is None:
        return
    processed_data = transform_for_fusion(df)
    csv_path = save_csv_for_training(processed_data,logger=logger,TRAINING_DIR=TRAINING_DIR)
    upload_to_ai_server(csv_path)
    mark_processed(df)
