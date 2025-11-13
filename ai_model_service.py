import io
import joblib
import time
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from logger.logger_singleton import getLogger
from utils.utils import save_csv_safely

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
    def __init__(self, df):
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
# Flatten and calculate predicted_return / predicted_hold_days
# -----------------------------
def flatten_entry_with_max_return(entry, min_relative_gain=0.01):
    flat_row = {
        "osiKey": entry["osiKey"],
        "strikePrice": entry["strikePrice"],
        "optionType": entry["optionType"],
        "moneyness": entry["moneyness"],
    }

    entry_price = entry["sequence"][0]["lastPrice"]
    max_profit = -float("inf")
    hold_days_at_max = 0

    for idx, step in enumerate(entry["sequence"]):
        for k, v in step.items():
            flat_row[f"{k}_{idx}"] = v
        profit = (step["lastPrice"] - entry_price) / entry_price
        if profit > max_profit:
            max_profit = profit
            hold_days_at_max = idx

    # Respect minimum relative gain threshold
    if max_profit < min_relative_gain:
        flat_row["predicted_return"] = 0.0
        flat_row["predicted_hold_days"] = 0
    else:
        flat_row["predicted_return"] = max_profit
        flat_row["predicted_hold_days"] = hold_days_at_max

    return flat_row

def save_csv_for_training(data, logger, TRAINING_DIR: Path):
    """
    Flatten and save lifetime sequences as CSV incrementally (no huge memory spike)
    """
    logger.logMessage("Saving CSV")
    cnt = 0
    skipped = 0
    chunk_size = 5000
    throttle_every = 1000
    delay = 0.05

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRAINING_DIR.mkdir(exist_ok=True)
    final_path = TRAINING_DIR / f"lifetime_training_{timestamp}.csv"

    with tempfile.NamedTemporaryFile("w", delete=False, dir=TRAINING_DIR, newline="") as tmp_file:
        writer = None
        batch = []

        for entry in data:
            cnt += 1
            try:
                flat_row = flatten_entry_with_max_return(entry)
                batch.append(flat_row)
            except Exception as e:
                skipped += 1
                logger.logMessage(f"⚠️ Skipped entry {cnt}: {e}")
                continue

            if len(batch) >= chunk_size:
                df = pd.DataFrame(batch)
                if writer is None:
                    df.to_csv(tmp_file, index=False)
                    writer = True
                else:
                    df.to_csv(tmp_file, index=False, header=False)
                tmp_file.flush()
                batch.clear()

            if cnt % 100 == 0:
                logger.logMessage(f"Processed {cnt} entries ({skipped} skipped)...")
            if cnt % throttle_every == 0:
                time.sleep(delay)

        if batch:
            df = pd.DataFrame(batch)
            if writer is None:
                df.to_csv(tmp_file, index=False)
            else:
                df.to_csv(tmp_file, index=False, header=False)
            tmp_file.flush()

    Path(tmp_file.name).replace(final_path)
    logger.logMessage(f"✅ Training CSV saved to {final_path} ({cnt-skipped} rows, {skipped} skipped)")
    return final_path.name

# -----------------------------
# Accumulate CSV
# -----------------------------
def append_accumulated_data(new_df: pd.DataFrame):
    TRAINING_DIR.mkdir(exist_ok=True)
    if ACCUMULATED_DATA_PATH.exists():
        old_df = pd.read_csv(ACCUMULATED_DATA_PATH)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    save_csv_safely(combined, ACCUMULATED_DATA_PATH, logger=logger)
    logger.logMessage(f"📈 Accumulated dataset now has {len(combined)} rows.")
    return combined

# -----------------------------
# Streamed Hybrid Training (VRAM & CPU safe)
# -----------------------------
def train_hybrid_model_streamed(csv_path: Path, batch_size=BATCH_SIZE, throttle_delay=0.05, device=DEVICE):
    all_rows = 0
    sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, warm_start=True)
    scaler = StandardScaler()
    transformer_model = OptionTransformer(len(FEATURE_COLUMNS), HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    transformer_model.train()

    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        chunk = chunk.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
        if chunk.empty:
            continue

        X_scaled = scaler.fit_transform(chunk[FEATURE_COLUMNS].values)
        y_return = chunk["predicted_return"].values
        y_hold = chunk["predicted_hold_days"].values
        sgd_model.partial_fit(X_scaled, y_return)

        dataset = OptionLifetimeDataset(chunk)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred_return, pred_hold = transformer_model(X_batch)
            loss = criterion(pred_return, y_batch[:,0].unsqueeze(1)) + \
                   criterion(pred_hold, y_batch[:,1].unsqueeze(1))
            loss.backward()
            optimizer.step()
            if throttle_delay > 0:
                time.sleep(throttle_delay)

        all_rows += len(chunk)

    hybrid_model = {"sgd": sgd_model, "transformer": transformer_model}
    save_model(hybrid_model, scaler)
    logger.logMessage(f"📊 Streamed hybrid training complete: {all_rows} rows processed")
    return {"status": "trained", "rows": all_rows}

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...), auto_train: bool = Form(default=False)):
    try:
        tmp_path = TRAINING_DIR / f"_upload_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(tmp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        # Read, process, and calculate predicted_return/hold
        processed_rows = []
        for chunk in pd.read_csv(tmp_path, chunksize=50_000):
            chunk = chunk.dropna().reset_index(drop=True)
            for _, row in chunk.iterrows():
                if "sequence" in row:
                    processed_rows.append(flatten_entry_with_max_return(row))
        tmp_path.unlink(missing_ok=True)

        # Save to final CSV
        csv_name = save_csv_for_training(processed_rows, logger, TRAINING_DIR)
        accumulated_df = append_accumulated_data(pd.read_csv(TRAINING_DIR / csv_name))

        result = {"status": "appended", "rows": len(accumulated_df)}

        if auto_train:
            stats = train_hybrid_model_streamed(TRAINING_DIR / csv_name)
            result.update(stats)

        return result

    except Exception as e:
        logger.logMessage(f"Upload error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/train")
def train_accumulated():
    if not ACCUMULATED_DATA_PATH.exists():
        return {"status": "error", "message": "No accumulated data found."}
    stats = train_hybrid_model_streamed(ACCUMULATED_DATA_PATH)
    return stats

@app.post("/predict")
async def predict(data: dict):
    features = data.get("features", {})
    if not features:
        return {"status": "error", "message": "No features provided."}

    model_dict, scaler = load_existing_model()
    if model_dict is None:
        return {"status": "error", "message": "Model not trained yet."}

    x = np.array([features.get(f, 0) for f in FEATURE_COLUMNS]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    sgd_pred_return = model_dict["sgd"].predict(x_scaled)[0]
    transformer_model = model_dict["transformer"].to(DEVICE).eval()
    with torch.no_grad():
        X_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        pred_return, pred_hold = transformer_model(X_tensor)
        transformer_pred_return = pred_return.item()
        transformer_pred_hold = pred_hold.item()

    final_return = (sgd_pred_return + transformer_pred_return)/2
    final_hold = transformer_pred_hold
    signal = "BUY" if final_return > 0.05 else "SELL" if final_return < -0.05 else "HOLD"

    return {
        "status": "ok",
        "result": {
            "predicted_return": float(final_return),
            "predicted_days_to_hold": float(final_hold),
            "signal": signal
        }
    }
