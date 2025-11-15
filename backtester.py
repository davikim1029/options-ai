# backtester.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
from evaluation import load_model_and_scaler

# Status file path for live progress updates
BACKTEST_STATUS_PATH = Path("model_store/backtest_status.json")

# -----------------------------
# Status Helpers
# -----------------------------
def safe_write_status(data: dict):
    BACKTEST_STATUS_PATH.parent.mkdir(exist_ok=True)
    BACKTEST_STATUS_PATH.write_text(json.dumps(data, indent=2))

def begin_backtest_status():
    safe_write_status({"status": "running", "progress": 0, "results": None})

def finalize_backtest_status(results: dict):
    safe_write_status({"status": "complete", "progress": 100, "results": results})

# -----------------------------
# Streaming/Batched Backtest
# -----------------------------
def run_backtest_streaming(csv_path: Path, batch_size: int = 128) -> dict:
    """
    Perform memory-conscious backtest on large CSV.
    Computes: total_options, buys, win_rate, avg_trade_return, total_profit.
    """
    begin_backtest_status()

    model, scaler = load_model_and_scaler()

    total_options = 0
    buys = 0
    wins = 0
    pnl_total = 0
    pnl_list = []

    # Streaming read
    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        required_cols = {"osiKey", "seq_features", "final_profit"}
        missing = required_cols - set(chunk.columns)
        if missing:
            finalize_backtest_status({"error": f"Missing columns: {missing}"})
            return {"error": f"Missing columns: {missing}"}

        for _, row in chunk.iterrows():
            total_options += 1
            seq_array = np.array(row["seq_features"], dtype=float).reshape(1, -1)
            seq_scaled = scaler.transform(seq_array)

            # Prediction returns [pred_return, pred_days]
            pred_return, pred_days = model.predict(seq_scaled)[0]

            should_buy = pred_return > 0.05
            if should_buy:
                buys += 1
                true_return = row["final_profit"]
                pnl_total += true_return
                pnl_list.append(true_return)
                if true_return > 0:
                    wins += 1

        # Update status after each chunk
        safe_write_status({
            "status": "running",
            "progress": int((total_options / 1_000_000)*100)  # approximate if known total not available
        })

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
