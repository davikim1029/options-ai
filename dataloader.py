from pipeline import DB_PATH, TRAINING_DIR, fetch_new_lifetimes
import pandas as pd
from pathlib import Path
import glob

# Features and target columns, same as used in your training
FEATURE_COLUMNS = [
    "optionType","strikePrice","lastPrice","bid","ask","bidSize","askSize",
    "volume","openInterest","nearPrice","inTheMoney","delta","gamma","theta",
    "vega","rho","iv","spread","midPrice","moneyness","daysToExpiration"
]

TARGET_COLUMNS = ["predicted_return", "predicted_hold_days"]

# -----------------------------
# Load all historical option lifetime data (DB)
# -----------------------------
def load_lifetime_dataset(min_rows=1_000):
    """Load completed option lifetimes from DB for backtesting/evaluation."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM option_lifetimes", conn)
    conn.close()

    if df.empty or len(df) < min_rows:
        raise RuntimeError(f"Not enough lifetime data in DB ({len(df)})")

    # Add totalSnapshots column if missing
    if "totalSnapshots" not in df.columns:
        snapshot_counts = df.groupby("osiKey").size().reset_index(name="totalSnapshots")
        df = df.merge(snapshot_counts, on="osiKey", how="left")

    return df

# -----------------------------
# Load all accumulated CSVs in training dir
# -----------------------------
def load_accumulated_training_csvs(chunk_size=25_000):
    """Load all CSVs from TRAINING_DIR in RAM-safe chunks."""
    all_files = sorted(Path(TRAINING_DIR).glob("*.csv"))
    if not all_files:
        raise RuntimeError(f"No training CSVs found in {TRAINING_DIR}")

    # Concatenate all CSVs into one DataFrame safely
    dfs = []
    for f in all_files:
        for chunk in pd.read_csv(f, chunksize=chunk_size):
            dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    return df
