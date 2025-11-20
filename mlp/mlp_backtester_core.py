# mlp_backtester_core.py
import sqlite3
import json
import traceback
from threading import Lock, get_ident
from pathlib import Path
import os
from shared_options.log.logger_singleton import getLogger
from utils.utils import load_model
from constants import DB_PATH, BATCH_SIZE, FEATURE_COLUMNS

BACKTEST_STATUS_PATH = Path("model_store/mlp_backtest_status.json")
_backtest_lock = Lock()
logger = getLogger()
LOG_EVERY_N = 500  # log every N rows


# -----------------------------
# Status helpers
# -----------------------------
def safe_write_status(data: dict):
    BACKTEST_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_STATUS_PATH.write_text(json.dumps(data, indent=2))


def begin_backtest_status():
    safe_write_status({"status": "running", "progress": 0, "results": None})


def finalize_backtest_status(results: dict):
    safe_write_status({"status": "complete", "progress": 100, "results": results})


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(row, feature_columns=FEATURE_COLUMNS):
    """Extract numeric feature vector for MLP."""
    row_dict = dict(row)
    return [float(row_dict.get(f, 0.0)) for f in feature_columns]


def compute_actual_outcome(row):
    """Compute realized PnL and hold period."""
    row_dict = dict(row)
    entry_price = float(row_dict.get("entryPrice", 0.0))
    exit_price = float(row_dict.get("exitPrice", entry_price))
    profit = exit_price - entry_price
    hold_days = int(row_dict.get("holdDays", 1))
    return profit, hold_days


# -----------------------------
# Streaming backtester
# -----------------------------
def run_backtest_permutations(db_path=DB_PATH, batch_size=BATCH_SIZE, fastapi_request=None):
    pid = os.getpid()
    tid = get_ident()
    req_id = getattr(fastapi_request.state, "request_id", "no-request") if fastapi_request else "no-request"
    logger.logMessage(f"[request_id={req_id}] PID={pid} Thread={tid} Entered MLP Backtester")

    acquired = _backtest_lock.acquire(blocking=False)
    if not acquired:
        return {"error": "Another backtest is already running."}

    conn = None
    try:
        begin_backtest_status()
        model, scaler, feature_columns = load_model()

        total_options = 0
        buys = 0
        wins = 0
        pnl_total = 0.0

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        total_rows = conn.execute("SELECT COUNT(*) FROM option_permutations").fetchone()[0]
        offset = 0

        while True:
            rows = conn.execute(
                f"SELECT * FROM option_permutations LIMIT {batch_size} OFFSET {offset}"
            ).fetchall()
            if not rows:
                break

            offset += batch_size

            for row in rows:
                total_options += 1
                try:
                    actual_profit, actual_hold = compute_actual_outcome(row)
                    feats = extract_features(row, feature_columns)
                    X_scaled = scaler.transform([feats])
                    pred_return, pred_hold = model.predict(X_scaled)[0]  # unpack

                    if pred_return > 0.05:  # buy threshold
                        buys += 1
                        pnl_total += actual_profit
                        if actual_profit > 0:
                            wins += 1

                except Exception as e:
                    logger.logMessage(f"⚠️ Skipping malformed row: {e}")
                    continue

                if total_options % LOG_EVERY_N == 0:
                    progress = int((total_options / total_rows) * 100)
                    safe_write_status({"status": "running", "progress": progress, "results": None})
                    logger.logMessage(f"[request_id={req_id}] Processed {total_options}/{total_rows} rows ({progress}%)")

        win_rate = (wins / buys) if buys else 0.0
        avg_return = (pnl_total / buys) if buys else 0.0

        results = {
            "total_options": total_options,
            "buys": buys,
            "win_rate": win_rate,
            "avg_trade_return": avg_return,
            "total_profit": pnl_total
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
        if conn:
            conn.close()
