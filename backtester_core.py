# backtester_core.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
from evaluation import load_model_and_scaler
from ast import literal_eval
from threading import Lock

# -----------------------------
# Config / Status
# -----------------------------
BACKTEST_STATUS_PATH = Path("model_store/backtest_status.json")
_backtest_lock = Lock()

# -----------------------------
# Status Helpers
# -----------------------------
def safe_write_status(data: dict):
    BACKTEST_STATUS_PATH.parent.mkdir(exist_ok=True, parents=True)
    BACKTEST_STATUS_PATH.write_text(json.dumps(data, indent=2))

def begin_backtest_status():
    safe_write_status({"status": "running", "progress": 0, "results": None})

def finalize_backtest_status(results: dict):
    safe_write_status({"status": "complete", "progress": 100, "results": results})

# -----------------------------
# Core Streaming Backtest
# -----------------------------
def run_backtest_streaming(csv_path: Path, batch_size: int = 128, total_rows: int = None) -> dict:
    """
    Perform memory-conscious backtest on large CSV.
    Computes: total_options, buys, win_rate, avg_trade_return, total_profit.
    """
    if not _backtest_lock.acquire(blocking=False):
        return {"error": "Another backtest is already running."}

    try:
        begin_backtest_status()
        model, scaler = load_model_and_scaler()

        total_options = 0
        buys = 0
        wins = 0
        pnl_total = 0

        # Attempt to estimate total rows if not given
        if total_rows is None:
            try:
                total_rows = sum(1 for _ in pd.read_csv(csv_path, usecols=[0], chunksize=100_000))
            except Exception:
                total_rows = None  # fallback

        for chunk in pd.read_csv(csv_path, chunksize=batch_size):
            required_cols = {"osiKey", "seq_features", "final_profit"}
            missing = required_cols - set(chunk.columns)
            if missing:
                finalize_backtest_status({"error": f"Missing columns: {missing}"})
                return {"error": f"Missing columns: {missing}"}

            # Vectorized parsing of seq_features
            seq_arrays = chunk["seq_features"].apply(lambda x: np.array(literal_eval(x), dtype=float))
            
            for idx, seq_array in enumerate(seq_arrays):
                total_options += 1
                seq_scaled = scaler.transform(seq_array.reshape(1, -1))
                pred_return, pred_days = model.predict(seq_scaled)[0]

                if pred_return > 0.05:
                    buys += 1
                    true_return = chunk.iloc[idx]["final_profit"]
                    pnl_total += true_return
                    if true_return > 0:
                        wins += 1

            # Update status per chunk
            progress = int((total_options / total_rows) * 100) if total_rows else 0
            safe_write_status({"status": "running", "progress": progress, "results": None})

        win_rate = (wins / buys) if buys else 0
        avg_return = (pnl_total / buys) if buys else 0

        results = {
            "total_options": total_options,
            "buys": buys,
            "win_rate": win_rate,
            "avg_trade_return": avg_return,
            "total_profit": pnl_total
        }

        finalize_backtest_status(results)
        return results
    finally:
        _backtest_lock.release()
