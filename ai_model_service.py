# ai_server.py
import io
import joblib
import time
import ijson  # streaming JSON parser
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from logger.logger_singleton import getLogger
from utils.utils import safe_literal_eval,to_native_types

logger = getLogger()
app = FastAPI(title="Hybrid AI Model Service")

# -----------------------------
# Paths & Constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"
MODEL_DIR = BASE_DIR / "models" / "versions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "current_model.pkl"
SCALER_PATH = MODEL_DIR / "current_scaler.pkl"
ACCUMULATED_DATA_PATH = TRAINING_DIR / "accumulated_training.csv"

# canonical feature columns expected by model (single-snapshot features)
FEATURE_COLUMNS = [
    "optionType","strikePrice","lastPrice","bid","ask","bidSize","askSize",
    "volume","openInterest","nearPrice","inTheMoney","delta","gamma","theta",
    "vega","rho","iv","spread","midPrice","moneyness","daysToExpiration"
]
TARGET_COLUMNS = ["predicted_return","predicted_hold_days"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2


# -----------------------------
# Dataset / Model
# -----------------------------
class OptionLifetimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Ensure FEATURE_COLUMNS and TARGET_COLUMNS exist
        for c in FEATURE_COLUMNS + TARGET_COLUMNS:
            if c not in df.columns:
                # fill missing features with zeros
                df[c] = 0.0 if c in TARGET_COLUMNS or df[c:c+1].select_dtypes(include=[np.floating]).any().any() else 0
        self.X = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.y = df[TARGET_COLUMNS].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class OptionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out[:,0].unsqueeze(1), out[:,1].unsqueeze(1)

# -----------------------------
# Load / Save
# -----------------------------
def load_existing_model():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

def save_model(model, scaler):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODEL_DIR / f"model_{timestamp}.pkl"
    scaler_file = MODEL_DIR / f"scaler_{timestamp}.pkl"
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    model_file.replace(MODEL_PATH)
    scaler_file.replace(SCALER_PATH)
    logger.logMessage(f"Saved hybrid model -> {model_file}")

# -----------------------------
# Flatten / label generation
# -----------------------------
def compute_max_profit_after_valley(sequence, min_expected_gain=0.01, min_risk_reward=0.1, smoothing_window=3, late_game_bump_threshold=0.02):
    """
    Given a sequence (list of dicts with 'lastPrice' and timestamp), compute a robust
    predicted_return and predicted_hold_days using valley/peak logic and filtering.
    Returns (predicted_return(float), predicted_hold_days(int)).
    """
    # validate
    if not isinstance(sequence, list) or len(sequence) == 0:
        return 0.0, 0

    prices = [float(x.get("lastPrice", 0.0)) for x in sequence]
    n = len(prices)

    # smoothing by taking local window max (reduces single-snapshot spikes)
    smoothed = []
    for i in range(n):
        window_end = min(n, i + smoothing_window)
        smoothed.append(max(prices[i:window_end]))

    # For each potential entry (but here we'll compute relative to the first snapshot, caller can iterate if needed)
    entry_price = prices[0]
    max_profit = -float("inf")
    hold_days_at_max = 0
    in_valley = False
    current_valley = entry_price
    min_future_price = entry_price

    # find peaks after valleys
    for j in range(0, n):
        p = smoothed[j]
        if p < current_valley:
            current_valley = p
            in_valley = True
        elif in_valley and p > entry_price:
            profit = (p - entry_price) / max(entry_price, 1e-9)
            if profit > max_profit:
                max_profit = profit
                hold_days_at_max = j
        if prices[j] < min_future_price:
            min_future_price = prices[j]

    # fallback: if no peak after valley, use absolute max from sequence
    if max_profit == -float("inf"):
        future_profits = [(p - entry_price) / max(entry_price, 1e-9) for p in prices]
        max_profit = max(future_profits)
        hold_days_at_max = future_profits.index(max_profit)

    # cap hold_days within lifetime
    hold_days_at_max = min(hold_days_at_max, n - 1)

    # risk/reward filter
    potential_loss = max(entry_price - min_future_price, 1e-9)
    risk_reward_ratio = max_profit / potential_loss if potential_loss > 0 else float("inf")

    predicted_return = float(max_profit) if (max_profit >= min_expected_gain and risk_reward_ratio >= min_risk_reward) else 0.0
    predicted_hold_days = int(hold_days_at_max) if predicted_return > 0 else 0

    # late-game bump filter
    remaining_days = n
    if remaining_days > 0 and predicted_hold_days > 0 and (predicted_hold_days / remaining_days) > 0.8 and predicted_return < late_game_bump_threshold:
        predicted_return = 0.0
        predicted_hold_days = 0

    return predicted_return, predicted_hold_days

def flatten_entry_for_training(entry, min_expected_gain=0.05, min_risk_reward=0.2, smoothing_window=3, late_game_bump_threshold=0.02):
    """
    Convert a raw option 'entry' (dict with static fields and 'sequence' list) into
    a *single-row* dict of features + targets. The model expects single-snapshot style
    features (FEATURE_COLUMNS) so we extract the first snapshot as canonical snapshot,
    plus the computed predicted_return/predicted_hold_days.
    """
    # Basic static fields
    flat = {}
    flat["osiKey"] = entry.get("osiKey")
    flat["strikePrice"] = float(entry.get("strikePrice", 0.0))
    flat["optionType"] = entry.get("optionType", "")
    # sequence must be list[dict]
    seq = entry.get("sequence")
    if isinstance(seq, str):
        seq = safe_literal_eval(seq) or []
    if not isinstance(seq, list):
        seq = []

    # If we have at least one snapshot, use first snapshot's values for core features
    first = seq[0] if seq else {}

    # Map FEATURE_COLUMNS from first snapshot and static fields
    # Provide safe defaults if missing
    flat["lastPrice"] = float(first.get("lastPrice", 0.0))
    flat["bid"] = float(first.get("bid", 0.0))
    flat["ask"] = float(first.get("ask", 0.0))
    flat["bidSize"] = float(first.get("bidSize", 0.0))
    flat["askSize"] = float(first.get("askSize", 0.0))
    flat["volume"] = float(first.get("volume", 0.0))
    flat["openInterest"] = float(first.get("openInterest", 0.0))
    flat["nearPrice"] = float(first.get("nearPrice", 0.0))
    flat["inTheMoney"] = float(first.get("inTheMoney", 0.0))
    flat["delta"] = float(first.get("delta", 0.0))
    flat["gamma"] = float(first.get("gamma", 0.0))
    flat["theta"] = float(first.get("theta", 0.0))
    flat["vega"] = float(first.get("vega", 0.0))
    flat["rho"] = float(first.get("rho", 0.0))
    flat["iv"] = float(first.get("iv", 0.0))
    flat["spread"] = float(first.get("spread", 0.0))
    flat["midPrice"] = float(first.get("midPrice", 0.0))
    flat["moneyness"] = float(first.get("moneyness", 0.0))
    flat["daysToExpiration"] = float(first.get("daysToExpiration", 0.0))

    # Compute robust targets
    predicted_return, predicted_hold_days = compute_max_profit_after_valley(
        seq,
        min_expected_gain=min_expected_gain,
        min_risk_reward=min_risk_reward,
        smoothing_window=smoothing_window,
        late_game_bump_threshold=late_game_bump_threshold
    )
    flat["predicted_return"] = float(predicted_return)
    flat["predicted_hold_days"] = int(predicted_hold_days)

    # Ensure native python types for JSON safety
    for k, v in list(flat.items()):
        if isinstance(v, (np.generic,)):
            flat[k] = v.item()

    return flat

# -----------------------------
# CSV helpers (streaming)
# -----------------------------
def save_rows_to_csv_stream(rows_iterable, out_path: Path, chunk_size: int = 5000):
    """
    Write rows (dicts) into CSV incrementally. rows_iterable yields dict rows.
    """
    TRAINING_DIR.mkdir(exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    header_written = out_path.exists()
    buffer = []
    written = 0
    for row in rows_iterable:
        # normalize types
        row = to_native_types(row)
        buffer.append(row)
        if len(buffer) >= chunk_size:
            df = pd.DataFrame(buffer)
            df.to_csv(tmp, mode="a", index=False, header=(not tmp.exists() and not header_written))
            written += len(buffer)
            buffer.clear()
    if buffer:
        df = pd.DataFrame(buffer)
        df.to_csv(tmp, mode="a", index=False, header=(not tmp.exists() and not header_written))
        written += len(buffer)
        buffer.clear()
    # If out_path does not exist yet and tmp exists, move
    if not out_path.exists() and tmp.exists():
        tmp.replace(out_path)
    else:
        # append tmp's lines to out_path (if both exist)
        if tmp.exists():
            with open(out_path, "a") as fa, open(tmp, "r") as ft:
                # skip header if exists
                first = True
                for line in ft:
                    if first:
                        # skip header line from tmp
                        first = False
                        continue
                    fa.write(line)
            tmp.unlink(missing_ok=True)
    return written

def append_accumulated_data_append_only(new_df: pd.DataFrame):
    """
    Append new_df to accumulated CSV using to_csv mode='a' to avoid loading full CSV.
    Returns number of rows appended and the path.
    """
    TRAINING_DIR.mkdir(exist_ok=True)
    new_df = new_df.copy()
    # ensure native types for saving
    for col in new_df.select_dtypes(include=[np.integer, np.floating]).columns:
        # cast numpy types to native
        if new_df[col].dtype.kind in ("i", "u"):
            new_df[col] = new_df[col].astype(np.int64).astype("int64")
        else:
            new_df[col] = new_df[col].astype("float64")
    mode = "a" if ACCUMULATED_DATA_PATH.exists() else "w"
    header = not ACCUMULATED_DATA_PATH.exists()
    new_df.to_csv(ACCUMULATED_DATA_PATH, mode=mode, index=False, header=header)
    # return the path (server works with path)
    total = sum(1 for _ in pd.read_csv(ACCUMULATED_DATA_PATH, chunksize=100_000))
    logger.logMessage(f"📈 Accumulated dataset now has {total} rows.")
    return ACCUMULATED_DATA_PATH, len(new_df)

# -----------------------------
# Streaming hybrid training (VRAM & CPU safe)
# -----------------------------
def train_hybrid_model_streamed(csv_path: Path, batch_size=BATCH_SIZE, throttle_delay=0.05, device=DEVICE):
    """
    Fully streaming hybrid training:
     - Incremental SGD via partial_fit
     - Transformer trained chunk-by-chunk (batches to GPU only)
     - Throttling between batches to keep server responsive
    """
    logger.logMessage("Starting streamed hybrid training...")
    sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, warm_start=True)
    scaler = StandardScaler()
    transformer_model = OptionTransformer(len(FEATURE_COLUMNS), HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    transformer_model.train()

    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        # ensure required columns exist; fill missing with zeros
        for c in FEATURE_COLUMNS + TARGET_COLUMNS:
            if c not in chunk.columns:
                chunk[c] = 0.0

        chunk = chunk.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
        if chunk.empty:
            continue

        X_chunk = chunk[FEATURE_COLUMNS].values.astype(np.float32)
        y_chunk = chunk[TARGET_COLUMNS].values.astype(np.float32)

        # scaler fit on first chunk, transform subsequently
        if first_chunk:
            X_scaled = scaler.fit_transform(X_chunk)
            first_chunk = False
        else:
            X_scaled = scaler.transform(X_chunk)

        # SGD incremental update on returns (target 0)
        sgd_model.partial_fit(X_scaled, y_chunk[:, 0])

        # Transformer training chunk-by-chunk
        dataset = OptionLifetimeDataset(chunk)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred_return, pred_hold = transformer_model(X_batch)
            loss = criterion(pred_return, y_batch[:,0].unsqueeze(1)) + criterion(pred_hold, y_batch[:,1].unsqueeze(1))
            loss.backward()
            optimizer.step()
            if throttle_delay > 0:
                time.sleep(throttle_delay)

        total_rows += len(chunk)

    # Save model & scaler
    hybrid_model = {"sgd": sgd_model, "transformer": transformer_model}
    save_model(hybrid_model, scaler)
    logger.logMessage(f"📊 Streamed hybrid training complete: {total_rows} rows processed")
    return jsonable_encoder({"status": "trained", "rows": int(total_rows)})

# -----------------------------
# Upload endpoint (CSV or JSON sequences)
# -----------------------------
@app.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...), auto_train: bool = Form(default=False)):
    """
    Stream a potentially huge JSON file to disk, process it item-by-item,
    flatten entries, and append to accumulated CSV. Logging shows progress.
    """
    try:
        TRAINING_DIR.mkdir(exist_ok=True)
        tmp_path = TRAINING_DIR / f"_upload_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 1️⃣ Save uploaded file to temp
        with open(tmp_path, "wb") as fw:
            while chunk := await file.read(1024 * 1024):
                fw.write(chunk)

        total_processed = 0
        total_skipped = 0

        def generator():
            nonlocal total_processed, total_skipped
            with open(tmp_path, "rb") as fr:
                # ijson.items allows streaming JSON arrays without loading all in memory
                for item in ijson.items(fr, "item"):
                    try:
                        flat_entry = flatten_entry_for_training(item)
                        total_processed += 1
                        if total_processed % 10000 == 0:
                            logger.logMessage(f"Processed {total_processed} items, skipped {total_skipped}")
                        yield flat_entry
                    except Exception as e:
                        total_skipped += 1
                        logger.logMessage(f"⚠️ Skipping malformed entry #{total_processed + total_skipped}: {e}")

        # 2️⃣ Save processed entries to CSV
        tmp_csv = TRAINING_DIR / f"_upload_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        written = save_rows_to_csv_stream(generator(), tmp_csv, chunk_size=5000)

        # 3️⃣ Append to accumulated dataset
        if written:
            df_chunk = pd.read_csv(tmp_csv)
            path, cnt = append_accumulated_data_append_only(df_chunk)
            total_processed = cnt
            tmp_csv.unlink(missing_ok=True)

        tmp_path.unlink(missing_ok=True)

        logger.logMessage(f"✅ Upload complete: {total_processed} entries processed, {total_skipped} skipped")
        
        result = {"status": "appended", "rows": int(total_processed), "skipped": int(total_skipped)}

        # 4️⃣ Optional: auto_train
        if auto_train:
            train_result = train_hybrid_model_streamed(
                ACCUMULATED_DATA_PATH,
                batch_size=BATCH_SIZE,
                throttle_delay=0.05,
                device=DEVICE
            )
            return train_result

        return result

    except Exception as e:
        logger.logMessage(f"Upload error: {e}")
        return {"status": "error", "message": str(e)}

# -----------------------------
# Train endpoint (train on accumulated)
# -----------------------------
@app.post("/train")
def train_accumulated():
    try:
        if not ACCUMULATED_DATA_PATH.exists():
            return jsonable_encoder({"status": "error", "message": "No accumulated data found."})
        result = train_hybrid_model_streamed(ACCUMULATED_DATA_PATH, batch_size=BATCH_SIZE, throttle_delay=0.05, device=DEVICE)
        return result
    except Exception as e:
        logger.logMessage(f"Training error: {e}")
        return jsonable_encoder({"status": "error", "message": str(e)})

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
async def predict(data: dict):
    try:
        features = data.get("features", {})
        if not features:
            return jsonable_encoder({"status": "error", "message": "No features provided."})

        model_dict, scaler = load_existing_model()
        if model_dict is None:
            return jsonable_encoder({"status": "error", "message": "Model not trained yet."})

        # Build feature vector (fill missing with 0)
        x = np.array([features.get(f, 0) for f in FEATURE_COLUMNS]).reshape(1, -1)
        x_scaled = scaler.transform(x)

        # SGD prediction
        sgd_pred_return = model_dict["sgd"].predict(x_scaled)[0]

        # Transformer prediction
        transformer_model = model_dict["transformer"].to(DEVICE).eval()
        with torch.no_grad():
            X_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            pred_return, pred_hold = transformer_model(X_tensor)
            transformer_pred_return = float(pred_return.item())
            transformer_pred_hold = float(pred_hold.item())

        final_return = float((sgd_pred_return + transformer_pred_return) / 2.0)
        final_hold = float(transformer_pred_hold)
        signal = "BUY" if final_return > 0.05 else "SELL" if final_return < -0.05 else "HOLD"

        return jsonable_encoder({
            "status": "ok",
            "result": {
                "predicted_return": final_return,
                "predicted_days_to_hold": final_hold,
                "signal": signal
            }
        })
    except Exception as e:
        logger.logMessage(f"Predict error: {e}")
        return jsonable_encoder({"status": "error", "message": str(e)})

