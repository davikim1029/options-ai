# ai_server.py  (FULL production-ready rewrite - drop-in replacement)
import uuid
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.encoders import jsonable_encoder
import torch
from torch import nn
from torch.utils.data import DataLoader
from logger.logger_singleton import getLogger
from utils.utils import safe_literal_eval, to_native_types
from evaluation import evaluate_model
from backtester_api import router as backtest_router
from upload_api import router as upload_router
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from starlette.middleware.base import BaseHTTPMiddleware
from training import load_existing_model,sequence_to_matrix,SequenceDataset,compute_end_of_life_targets
from constants import TRAINING_DIR,FEATURE_COLUMNS,TARGET_COLUMNS,MAX_SEQ_LEN_CAP,LOG_EVERY_N,DEVICE,BATCH_SIZE,HIDDEN_DIM,NUM_LAYERS,LEARNING_RATE,ACCUMULATED_DATA_PATH

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


logger = getLogger()
app = FastAPI(title="Hybrid AI Model Service (Seq Transformer)")
app.include_router(backtest_router)
app.include_router(upload_router)
app.add_middleware(RequestIDMiddleware)

# -----------------------------
# Predict helper (single snapshot or with optional sequence)
# -----------------------------
def predict_option(features: dict = None, sequence: list = None):
    """
    Predict using current hybrid model.
    - features: dict of FEATURE_COLUMNS (single snapshot)
    - sequence: optional list of snapshots (chronological). If provided, used directly (padded/truncated).
    If sequence absent, we repeat features into a padded sequence.
    """
    model_dict, scaler = load_existing_model()
    if model_dict is None:
        return {"status": "error", "message": "Model not trained yet."}
    sgd = model_dict["sgd"]
    transformer = model_dict["transformer"].to(DEVICE).eval()

    # ensure features exists
    if features is None and (not sequence or len(sequence) == 0):
        return {"status": "error", "message": "No features or sequence provided."}
    if features is None and sequence:
        features = sequence[0]
    # build X_flat
    x_flat = np.array([float(features.get(f, 0) or 0.0) for f in FEATURE_COLUMNS], dtype=np.float32).reshape(1, -1)
    # load scaler
    _, scaler = load_existing_model()
    if scaler is None:
        return {"status": "error", "message": "Scaler not found."}
    x_scaled = scaler.transform(x_flat)
    sgd_pred = float(sgd.predict(x_scaled)[0])

    # build sequence matrix
    # derive max_seq_len from transformer
    max_seq_len = getattr(transformer, "max_seq_len", min(64, MAX_SEQ_LEN_CAP))
    if sequence and isinstance(sequence, list) and len(sequence) > 0:
        seq = sequence
    else:
        # repeat the single snapshot to form a pseudo-sequence
        seq = [features] * max_seq_len
    seq_len = min(len(seq), max_seq_len)
    seq_mat = sequence_to_matrix(seq, max_seq_len)  # returns (max_seq_len, F)
    X_tensor = torch.tensor(seq_mat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        tr_pred, th_pred = transformer(X_tensor)
        tr_val = float(tr_pred.item())
        th_val = float(th_pred.item())

    final_return = float((sgd_pred + tr_val) / 2.0)
    final_hold = float(th_val)
    signal = "BUY" if final_return > 0.05 else "SELL" if final_return < -0.05 else "HOLD"

    return {
        "status": "ok",
        "result": {
            "predicted_return": final_return,
            "predicted_days_to_hold": final_hold,
            "signal": signal,
            "sgd_return": sgd_pred,
            "transformer_return": tr_val,
            "transformer_hold": th_val
        }
    }

# -----------------------------
# Evaluation & backtesting endpoints (streaming)
# -----------------------------
def evaluate_on_csv(csv_path: Path, batch_size: int = 128):
    model_dict, scaler = load_existing_model()
    if model_dict is None:
        return {"status": "error", "message": "Model not trained yet."}
    transformer = model_dict["transformer"].to(DEVICE).eval()

    mse_list = []
    mse_hold_list = []
    tp = fp = tn = fn = 0
    total = 0

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        for c in FEATURE_COLUMNS + TARGET_COLUMNS + ["sequence"]:
            if c not in chunk.columns:
                chunk[c] = 0.0
        chunk = chunk.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
        if chunk.empty:
            continue
        ds = SequenceDataset(chunk, getattr(transformer, "max_seq_len", min(64, MAX_SEQ_LEN_CAP)))
        dl = DataLoader(ds, batch_size=batch_size)
        with torch.no_grad():
            for X_batch, y_batch in dl:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pr, ph = transformer(X_batch)
                pr = pr.detach().cpu().numpy().astype(float)
                ph = ph.detach().cpu().numpy().astype(float)
                tr = y_batch[:,0].cpu().numpy().astype(float)
                th = y_batch[:,1].cpu().numpy().astype(float)
                mse_list.append(((pr - tr) ** 2).mean())
                mse_hold_list.append(((ph - th) ** 2).mean())
                pred_up = pr > 0.0
                true_up = tr > 0.0
                for pu, tu in zip(pred_up, true_up):
                    if pu and tu: tp += 1
                    elif pu and not tu: fp += 1
                    elif not pu and not tu: tn += 1
                    elif not pu and tu: fn += 1
                total += len(pr)
    mse_return = float(np.mean(mse_list)) if mse_list else None
    mse_hold = float(np.mean(mse_hold_list)) if mse_hold_list else None
    rmse_return = float(math.sqrt(mse_return)) if mse_return is not None else None
    rmse_hold = float(math.sqrt(mse_hold)) if mse_hold is not None else None
    accuracy = (tp + tn) / total if total > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None

    return {
        "status": "ok",
        "rows_evaluated": int(total),
        "mse_return": mse_return,
        "mse_hold": mse_hold,
        "rmse_return": rmse_return,
        "rmse_hold": rmse_hold,
        "directional_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }

def backtest_on_csv(csv_path: Path, entry_threshold: float = 0.05, capital_per_trade: float = 1000.0):
    model_dict, scaler = load_existing_model()
    if model_dict is None:
        return {"status": "error", "message": "Model not trained yet."}
    transformer = model_dict["transformer"].to(DEVICE).eval()

    pnl_list = []; returns_list = []; wins = 0; losses = 0; trades = 0

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        if "sequence" not in chunk.columns:
            logger.logMessage("Backtest requires 'sequence' column; skipping chunk")
            continue
        for _, row in chunk.iterrows():
            try:
                seq_raw = row["sequence"]
                if isinstance(seq_raw, str):
                    seq = safe_literal_eval(seq_raw) or json.loads(seq_raw)
                elif isinstance(seq_raw, list):
                    seq = seq_raw
                else:
                    seq = []
                if not seq:
                    continue
                # prepare tensor
                max_seq_len = getattr(transformer, "max_seq_len", min(64, MAX_SEQ_LEN_CAP))
                mat = sequence_to_matrix(seq, max_seq_len)
                X_tensor = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pr, ph = transformer(X_tensor)
                    pred = float(pr.item())
                actual_ret, actual_hold = compute_end_of_life_targets(seq)
                if pred >= entry_threshold:
                    profit = capital_per_trade * actual_ret
                    pnl_list.append(profit)
                    returns_list.append(actual_ret)
                    trades += 1
                    if profit > 0: wins += 1
                    else: losses += 1
            except Exception as e:
                logger.logMessage(f"⚠️ Backtest skipping malformed row: {e}")
                continue

    if trades == 0:
        return {"status": "ok", "trades": 0, "message": "No trades triggered."}

    total_pnl = float(np.sum(pnl_list))
    avg_return = float(np.mean(returns_list))
    win_rate = float(wins / trades)
    profit_factor = (sum(p for p in pnl_list if p > 0) / abs(sum(p for p in pnl_list if p < 0))) if any(p < 0 for p in pnl_list) else float("inf")
    returns_arr = np.array(pnl_list)
    sharpe = float(returns_arr.mean() / (returns_arr.std() + 1e-9)) if returns_arr.size > 1 else None
    cum = np.cumsum(returns_arr)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_drawdown = float(np.max(drawdown)) if drawdown.size > 0 else 0.0

    return {
        "status": "ok",
        "trades": int(trades),
        "total_pnl": total_pnl,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }

# -----------------------------
# Endpoints for evaluate/confusion/backtest/predict_one
# -----------------------------

@app.get("/confusion")
def confusion_endpoint():
    try:
        if not ACCUMULATED_DATA_PATH.exists():
            return jsonable_encoder(to_native_types({"status": "error", "message": "No accumulated data found."}))
        metrics = evaluate_on_csv(ACCUMULATED_DATA_PATH, batch_size=128)
        return jsonable_encoder(to_native_types(metrics.get("confusion", {})))
    except Exception as e:
        logger.logMessage(f"Confusion error: {e}")
        return jsonable_encoder(to_native_types({"status": "error", "message": str(e)}))

@app.post("/predict_one")
async def predict_one(feature: dict = Body(...)):
    try:
        # normalize to FEATURE_COLUMNS
        features = {f: feature.get(f, 0) for f in FEATURE_COLUMNS}
        sequence = feature.get("sequence", None)
        res = predict_option(features=features, sequence=sequence)
        return jsonable_encoder(to_native_types(res))
    except Exception as e:
        logger.logMessage(f"Predict_one error: {e}")
        return jsonable_encoder(to_native_types({"status": "error", "message": str(e)}))


@app.get("/evaluate")
def evaluate_endpoint(batch_size: int = 128):
    try:
        df = load_accumulated_training_csvs(chunk_size=batch_size)
        metrics = evaluate_model(df, batch_size=batch_size)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.logMessage(f"Evaluate error: {e}")
        return {"status": "error", "message": str(e)}

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
