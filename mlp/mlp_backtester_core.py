# mlp_backtester_core.py
import sqlite3
from pathlib import Path
import json
import traceback
from threading import Lock, get_ident
from shared_options.log.logger_singleton import getLogger
from utils.utils import load_model  
import numpy as np
import os

BACKTEST_STATUS_PATH = Path("model_store/mlp_backtest_status.json")
_backtest_lock = Lock()
logger = getLogger()
LOG_EVERY_N = 500  # log every N rows


# -----------------------------
# Status Helpers
# -----------------------------
def safe_write_status(data: dict):
    BACKTEST_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_STATUS_PATH.write_text(json.dumps(data, indent=2))


def begin_backtest_status():
    safe_write_status({"status": "running", "progress": 0, "results": None})


def finalize_backtest_status(results: dict):
    safe_write_status({"status": "complete", "progress": 100, "results": results})


# -----------------------------
# Helper Functions
# -----------------------------
def compute_actual_outcome(row):
    """Compute realized PnL and hold period from a permutation row."""
    try:
        entry_price = float(row.get("entryPrice", 0.0))
        exit_price = float(row.get("exitPrice", entry_price))
        profit = exit_price - entry_price
        hold_days = int(row.get("holdDays", 1))
        return profit, hold_days
    except Exception:
        return 0.0, 0


def extract_features(row, feature_columns):
    """Extract numeric feature vector for MLP from a row."""
    feats = []
    for col in feature_columns:
        try:
            feats.append(float(row.get(col, 0.0)))
        except Exception:
            feats.append(0.0)
    return feats


# -----------------------------
# Streaming Backtester for MLP
# -----------------------------
def run_backtest_permutations(db_path: Path, batch_size: int = 128, fastapi_request=None):
    """
    Streaming backtester for MLP trained on option_permutations.
    Reads DB in chunks, computes realized vs predicted PnL/hold days.
    """
    pid = os.getpid()
    tid = get_ident()
    req_id = getattr(fastapi_request.state, "request_id", "no-request") if fastapi_request else "no-request"

    logger.logMessage(f"[request_id={req_id}] PID={pid} Thread={tid} Entered MLP Backtester")

    # Acquire lock
    acquired = _backtest_lock.acquire(blocking=False)
    if not acquired:
        return {"error": "Another backtest is already running."}

    try:
        begin_backtest_status()
        model, scaler, feature_columns = load_model()

        total_options = 0
        buys = 0
        wins = 0
        pnl_total = 0.0

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Determine total rows for progress
        try:
            total_rows = conn.execute("SELECT COUNT(*) FROM option_permutations").fetchone()[0]
        except:
            total_rows = None

        offset = 0
        chunk_index = 0

        while True:
            rows = conn.execute(
                f"SELECT * FROM option_permutations LIMIT {batch_size} OFFSET {offset}"
            ).fetchall()
            if not rows:
                break

            chunk_index += 1

            for row in rows:
                row = dict(row)
                total_options += 1

                try:
                    actual_profit, actual_hold = compute_actual_outcome(row)
                    feats = extract_features(row, feature_columns)
                    X = scaler.transform([feats])
                    pred_return, pred_hold = model.predict(X)[0]

                    if pred_return > 0.05:  # Buy threshold
                        buys += 1
                        pnl_total += actual_profit
                        if actual_profit > 0:
                            wins += 1

                except Exception as e:
                    logger.logMessage(f"⚠️ Skipping malformed row during backtest: {e}")
                    continue

                # Log progress every LOG_EVERY_N rows
                if total_options % LOG_EVERY_N == 0:
                    progress = int((total_options / total_rows) * 100) if total_rows else 0
                    safe_write_status({"status": "running", "progress": progress, "results": None})
                    logger.logMessage(f"[request_id={req_id}] Processed {total_options} rows, progress {progress}%")

            offset += batch_size

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
        logger.logMessage(f"[request_id={req_id}] Backtest complete: {results}")
        return results

    except Exception as e:
        logger.logMessage(f"[request_id={req_id}] MLP Backtest Exception: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

    finally:
        if acquired and _backtest_lock.locked():
            _backtest_lock.release()
            logger.logMessage(f"[request_id={req_id}] Released backtest lock")
        conn.close()
