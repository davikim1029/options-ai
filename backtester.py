# backtester.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
from evaluation import load_model_and_scaler
from ai_model_service import app,ACCUMULATED_DATA_PATH


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


@app.post("/backtest/run")
def api_backtest(batch_size: int = 256):
    """
    Launch a streaming, low-memory backtest.
    Returns immediately with basic info; progress stored in backtest_status.json.
    """
    if not ACCUMULATED_DATA_PATH.exists():
        return {"status": "error", "message": "No accumulated training dataset found."}

    # Kick off the backtest synchronously for now
    results = run_backtest_streaming(ACCUMULATED_DATA_PATH, batch_size=batch_size)
    return {"status": "complete", "results": results}


# ai_model_service/backtest_api.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import json
from threading import Lock
from backtester import run_backtest_streaming, BACKTEST_STATUS_PATH

class BacktestRequest(BaseModel):
    csv_path: str = "training/accumulated_training.csv"
    batch_size: int = 128

# Thread-safety lock to prevent multiple backtests at once
_backtest_lock = Lock()

@app.post("/backtest/start")
def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Start a new backtest in the background.
    """
    if not _backtest_lock.acquire(blocking=False):
        return {"status": "error", "message": "A backtest is already running."}

    csv_file = Path(request.csv_path)
    if not csv_file.exists():
        _backtest_lock.release()
        return {"status": "error", "message": f"CSV file not found: {csv_file}"}

    # Run backtest in background
    def _run():
        try:
            run_backtest_streaming(csv_file, batch_size=request.batch_size)
        finally:
            _backtest_lock.release()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Backtest running in background."}

@app.get("/backtest/status")
def get_backtest_status():
    """
    Get current backtest progress and results.
    """
    if not BACKTEST_STATUS_PATH.exists():
        return {"status": "idle", "progress": 0, "results": None}
    
    with BACKTEST_STATUS_PATH.open("r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"status": "error", "progress": 0, "results": None}
    
    return data

