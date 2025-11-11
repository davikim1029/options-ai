import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import random
from logger.logger_singleton import getLogger
import sqlite3
import os


logger = getLogger()

def generate_sample_training_file(output_dir: Path, n_rows: int = 500) -> Path:
    """
    Generates a sample CSV file of option data suitable for model training.
    The structure matches what the AI model expects in training uploads.

    Args:
        output_dir (Path): Directory where the file will be written.
        n_rows (int): Number of sample rows to generate.

    Returns:
        Path: The path of the created CSV file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"sample_option_training_{timestamp}.csv"

    np.random.seed(42)
    random.seed(42)

    option_types = [0, 1]  # 0 = put, 1 = call

    # Synthetic numeric distributions that mimic realistic option data
    df = pd.DataFrame({
        "optionType": np.random.choice(option_types, n_rows),
        "strikePrice": np.random.uniform(10, 200, n_rows),
        "lastPrice": np.random.uniform(5, 220, n_rows),
        "bid": np.random.uniform(4, 219, n_rows),
        "ask": np.random.uniform(5, 221, n_rows),
        "bidSize": np.random.randint(1, 100, n_rows),
        "askSize": np.random.randint(1, 100, n_rows),
        "volume": np.random.randint(1, 10000, n_rows),
        "openInterest": np.random.randint(10, 10000, n_rows),
        "nearPrice": np.random.uniform(10, 200, n_rows),
        "inTheMoney": np.random.choice([0, 1], n_rows),
        "delta": np.random.uniform(-1, 1, n_rows),
        "gamma": np.random.uniform(0, 0.2, n_rows),
        "theta": np.random.uniform(-0.1, 0.0, n_rows),
        "vega": np.random.uniform(0, 0.5, n_rows),
        "rho": np.random.uniform(-0.1, 0.1, n_rows),
        "iv": np.random.uniform(0.05, 1.0, n_rows),
        "spread": np.random.uniform(0.01, 2.0, n_rows),
        "midPrice": np.random.uniform(5, 220, n_rows),
        "moneyness": np.random.uniform(0.8, 1.2, n_rows),
        "daysToExpiration": np.random.uniform(1, 90, n_rows),
    })

    # Target variable: a numeric value between -1 (sell) and +1 (buy)
    # representing "recommendation strength"
    # +1 = strong buy, 0 = hold, -1 = strong sell
    df["recommendation"] = np.where(
        df["delta"] > 0.5,
        np.random.uniform(0.5, 1.0, n_rows),
        np.random.uniform(-1.0, 0.0, n_rows)
    )

    # Derived field: expected hold days (integer)
    df["expectedHoldDays"] = np.clip(
        np.random.normal(10, 5, n_rows).astype(int), 1, 30
    )

    df.to_csv(file_path, index=False)
    logger.logMessage(f"✅ Generated sample training data: {file_path} ({n_rows} rows)")
    return file_path



def copy_all_snapshot_data():
    # Current file directory
    here = os.path.dirname(os.path.abspath(__file__))

    # Go up one level (from options-alert to options) and then into option-file-server
    db_path = os.path.join(here,"..", "..", "option-file-server", "database", "options.db")

    DB_PATH = os.path.normpath(db_path)
    insert_sql = """
    INSERT INTO option_lifetimes (
        osiKey, timestamp, symbol, optionType, strikePrice, lastPrice,
        bid, ask, bidSize, askSize, volume, openInterest, nearPrice,
        inTheMoney, delta, gamma, theta, vega, rho, iv,
        daysToExpiration, spread, midPrice, moneyness, processed
    )
    SELECT
        osiKey, timestamp, symbol, optionType, strikePrice, lastPrice,
        bid, ask, bidSize, askSize, volume, openInterest, nearPrice,
        inTheMoney, delta, gamma, theta, vega, rho, iv,
        daysToExpiration, spread, midPrice, moneyness, 0
    FROM option_snapshots;
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Count rows before insert
        cur.execute("SELECT COUNT(*) FROM option_lifetimes;")
        before_count = cur.fetchone()[0]

        # Execute the insert
        cur.execute(insert_sql)
        conn.commit()

        # Count rows after insert
        cur.execute("SELECT COUNT(*) FROM option_lifetimes;")
        after_count = cur.fetchone()[0]

        rows_inserted = after_count - before_count
        print(f"Insert successful! {rows_inserted} rows added.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Quick manual test
    out_dir = Path("training")
    generate_sample_training_file(out_dir)
