# sgd_backtester_core.py

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from utils.utils import load_model
from ast import literal_eval
from threading import Lock, get_ident
from shared_options.log.logger_singleton import getLogger
from fastapi import Request
import traceback
from constants import FEATURE_COLUMNS,TrainerType

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
# Helpers
# -----------------------------
def compute_actual_outcome(sequence):
    """Compute realized profit and realized hold period from uploaded sequence."""
    try:
        if not isinstance(sequence, list) or len(sequence) < 2:
            return 0.0, 0

        entry = float(sequence[0].get("midPrice", 0.0))
        exit = float(sequence[-1].get("midPrice", 0.0))
        profit = exit - entry
        hold_days = len(sequence)

        return profit, hold_days

    except Exception:
        return 0.0, 0


def extract_features(row):
    """Extract scalar model features from row using EVAL_FEATURES."""
    feats = []
    for col in FEATURE_COLUMNS:
        try:
            feats.append(float(row.get(col, 0.0)))
        except:
            feats.append(0.0)
    return feats


# -----------------------------
# NEW Streaming Backtester
# -----------------------------
def run_backtest_streaming(csv_path: Path, batch_size: int = 128, total_rows: int = None, fastapi_request: Request = None):
    """
    NEW BACKTESTER:
    - Works with your actual training data format (sequence + scalar features)
    - Computes realized returns from the sequence field
    - Evaluates model predictions vs actual outcomes
    - Reports win rate / average return / total PnL
    - Streaming memory safe
    """

    logger = getLogger()
    pid = os.getpid()
    tid = get_ident()
    req_id = getattr(fastapi_request.state, "request_id", "no-request") if fastapi_request else "no-request"

    logger.logMessage(f"[request_id={req_id}] PID={pid} Thread={tid} Entered Run_Backtest Streaming")

    # Acquire lock
    acquired = _backtest_lock.acquire(blocking=False)
    if not acquired:
        return {"error": "Another backtest is already running."}

    try:
        logger.logMessage(f"[request_id={req_id}] Starting Backtest Streaming")
        begin_backtest_status()

        model, scaler = load_model(TrainerType.SGD)

        total_options = 0
        buys = 0
        wins = 0
        pnl_total = 0.0

        # Estimate rows if needed
        if total_rows is None:
            try:
                total_rows = sum(1 for _ in pd.read_csv(csv_path, usecols=[0], chunksize=100_000))
            except:
                total_rows = None

        chunk_index = 0
        for chunk in pd.read_csv(csv_path, chunksize=batch_size):
            chunk_index += 1

            # Required columns for this backtester
            required = {"sequence"} | set(FEATURE_COLUMNS)
            missing = required - set(chunk.columns)
            if missing:
                err = f"CSV missing required columns: {missing}"
                finalize_backtest_status({"error": err})
                return {"error": err}

            # Convert sequence column from JSON text -> Python list
            try:
                chunk["sequence"] = chunk["sequence"].apply(literal_eval)
            except Exception as e:
                err = f"Failed parsing sequence column: {e}"
                logger.logMessage(f"[request_id={req_id}] {err}")
                return {"error": err}

            # Process each row
            for _, row in chunk.iterrows():

                total_options += 1

                # Actual realized outcome from sequence
                actual_profit, actual_hold_days = compute_actual_outcome(row["sequence"])

                # Feature vector for prediction
                feats = extract_features(row)
                X = scaler.transform([feats])

                # Model prediction
                pred_return, pred_hold_days = model.predict(X)[0]

                # BUY rule: predicted_return > 5% (same as before)
                if pred_return > 0.05:
                    buys += 1
                    pnl_total += actual_profit
                    if actual_profit > 0:
                        wins += 1

            # Status update
            progress = int((total_options / total_rows) * 100) if total_rows else 0
            safe_write_status({"status": "running", "progress": progress, "results": None})

            if chunk_index % 50 == 0:
                logger.logMessage(f"[request_id={req_id}] Processed {chunk_index} chunks")

        # Final metrics
        win_rate = (wins / buys) if buys else 0.0
        avg_return = (pnl_total / buys) if buys else 0.0

        results = {
            "total_options": total_options,
            "buys": buys,
            "win_rate": win_rate,
            "avg_trade_return": avg_return,
            "total_profit": pnl_total,
        }

        finalize_backtest_status(results)
        return results

    except Exception as e:
        logger.logMessage(
            f"[request_id={req_id}] Backtest Exception: {e}\n{traceback.format_exc()}"
        )
        return {"error": str(e)}

    finally:
        if acquired and _backtest_lock.locked():
            _backtest_lock.release()
            logger.logMessage(f"[request_id={req_id}] Released backtest lock")
