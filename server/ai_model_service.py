# ai_server.py  (Refactored drop-in)
import uuid
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Body, APIRouter
from fastapi.encoders import jsonable_encoder
from starlette.middleware.base import BaseHTTPMiddleware
import torch
from torch.utils.data import DataLoader
import contextlib
from logger.logger_singleton import getLogger
from utils.utils import safe_literal_eval, to_native_types,load_model,TrainerType
from constants import (
    TRAINING_DIR, FEATURE_COLUMNS, TARGET_COLUMNS, MAX_SEQ_LEN_CAP,
    DEVICE, ACCUMULATED_DATA_PATH
)
from sgd.sgd_training import (
    sequence_to_matrix, SequenceDataset,
    compute_end_of_life_targets
)

# Include SGD + MLP routers
from sgd.sgd_backtester_api import router as sgd_backtest_router
from sgd.sgd_upload_api import router as sgd_upload_router
from mlp.mlp_backtester_api import router as mlp_backtest_router
from mlp.mlp_upload_api import router as mlp_upload_router

logger = getLogger()

# -----------------------------
# Middleware for Request IDs
# -----------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="Option AI Model Service")
app.add_middleware(RequestIDMiddleware)

# Include all routers
app.include_router(sgd_backtest_router)
app.include_router(sgd_upload_router)
app.include_router(mlp_backtest_router)
app.include_router(mlp_upload_router)


def sequence_to_tensor(sequence: list, max_seq_len: int):
    seq_len = min(len(sequence), max_seq_len)
    seq_mat = sequence_to_matrix(sequence, max_seq_len)
    return torch.tensor(seq_mat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

def predict_single(features: dict = None, sequence: list = None, modelType:TrainerType = TrainerType.MLP):
    try:
        model_dict, scaler = load_model(modelType)
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

    # Single snapshot
    if features is None and (not sequence or len(sequence) == 0):
        return {"status": "error", "message": "No features or sequence provided."}
    if features is None and sequence:
        features = sequence[0]

    x_flat = np.array([float(features.get(f, 0) or 0.0) for f in FEATURE_COLUMNS], dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x_flat) if scaler else x_flat

    sgd_pred = float(model_dict["sgd"].predict(x_scaled)[0]) if "sgd" in model_dict else 0.0

    transformer = model_dict.get("transformer", None)
    max_seq_len = getattr(transformer, "max_seq_len", min(64, MAX_SEQ_LEN_CAP)) if transformer else len(sequence or [features])

    seq_to_use = sequence if sequence and len(sequence) > 0 else [features] * max_seq_len
    X_tensor = sequence_to_tensor(seq_to_use, max_seq_len) if transformer else None

    if transformer:
        transformer.eval()
        with torch.no_grad():
            tr_pred, th_pred = transformer(X_tensor)
            tr_val = float(tr_pred.item())
            th_val = float(th_pred.item())
    else:
        tr_val = 0.0
        th_val = 0.0

    final_return = float((sgd_pred + tr_val) / 2.0) if transformer and "sgd" in model_dict else float(sgd_pred + tr_val)
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
# CSV Evaluation / Backtesting
# -----------------------------
def evaluate_on_csv(csv_path: Path, batch_size: int = 128, model_type:TrainerType = TrainerType.MLP):
    model_dict, scaler = load_model(model_type)
    transformer = model_dict.get("transformer", None)

    mse_list = []; mse_hold_list = []
    tp = fp = tn = fn = 0
    total = 0

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        for c in FEATURE_COLUMNS + TARGET_COLUMNS + ["sequence"]:
            if c not in chunk.columns:
                chunk[c] = 0.0
        chunk = chunk.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
        if chunk.empty:
            continue

        ds = SequenceDataset(chunk, getattr(transformer, "max_seq_len", min(64, MAX_SEQ_LEN_CAP))) if transformer else None
        dl = DataLoader(ds, batch_size=batch_size) if ds else [([chunk[FEATURE_COLUMNS].to_numpy()], [chunk[TARGET_COLUMNS].to_numpy()])]

        with torch.no_grad():
            for batch in dl:
                if transformer:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    pr, ph = transformer(X_batch)
                    pr, ph = pr.cpu().numpy(), ph.cpu().numpy()
                    tr, th = y_batch[:,0].cpu().numpy(), y_batch[:,1].cpu().numpy()
                else:
                    X, y = batch
                    pr, ph = np.zeros_like(y[:,0]), np.zeros_like(y[:,1])
                    tr, th = y[:,0], y[:,1]

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
    rmse_return = math.sqrt(mse_return) if mse_return is not None else None
    rmse_hold = math.sqrt(mse_hold) if mse_hold is not None else None
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

def backtest_on_csv(csv_path: Path, entry_threshold: float = 0.05, capital_per_trade: float = 1000.0, model_type:TrainerType = TrainerType.MLP):
    model_dict, scaler = load_model(model_type)
    transformer = model_dict.get("transformer", None)

    pnl_list = []; returns_list = []; wins = 0; losses = 0; trades = 0

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        if "sequence" not in chunk.columns:
            logger.logMessage(f"{model_type} backtest requires 'sequence' column; skipping chunk")
            continue
        for _, row in chunk.iterrows():
            try:
                seq_raw = row["sequence"]
                seq = safe_literal_eval(seq_raw) or json.loads(seq_raw) if isinstance(seq_raw, str) else seq_raw
                if not seq: continue
                max_seq_len = getattr(transformer, "max_seq_len", min(64, MAX_SEQ_LEN_CAP))
                X_tensor = sequence_to_tensor(seq, max_seq_len) if transformer else None

                with torch.no_grad() if transformer else contextlib.nullcontext():
                    pr, _ = transformer(X_tensor) if transformer else (0.0, 0.0)
                    pred = float(pr.item() if transformer else pr)

                actual_ret, _ = compute_end_of_life_targets(seq)
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
# Endpoints
# -----------------------------
@app.get("/confusion")
def confusion_endpoint():
    try:
        if not ACCUMULATED_DATA_PATH.exists():
            return jsonable_encoder({"status": "error", "message": "No accumulated data found."})
        metrics = evaluate_on_csv(ACCUMULATED_DATA_PATH, batch_size=128)
        return jsonable_encoder(to_native_types(metrics.get("confusion", {})))
    except Exception as e:
        logger.logMessage(f"Confusion error: {e}")
        return jsonable_encoder({"status": "error", "message": str(e)})

@app.post("/predict_one")
async def predict_one(feature: dict = Body(...), model_type: str = Body("hybrid")):
    try:
        features = {f: feature.get(f, 0) for f in FEATURE_COLUMNS}
        sequence = feature.get("sequence", None)
        res = predict_single(features=features, sequence=sequence, model_type=model_type)
        return jsonable_encoder(to_native_types(res))
    except Exception as e:
        logger.logMessage(f"Predict_one error: {e}")
        return jsonable_encoder({"status": "error", "message": str(e)})

# -----------------------------
# CSV Loader
# -----------------------------
def load_accumulated_training_csvs(chunk_size=25_000):
    all_files = sorted(Path(TRAINING_DIR).glob("*.csv"))
    if not all_files:
        raise RuntimeError(f"No training CSVs found in {TRAINING_DIR}")

    dfs = []
    for f in all_files:
        for chunk in pd.read_csv(f, chunksize=chunk_size):
            dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    return df
