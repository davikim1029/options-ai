import io
import joblib
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
# Transformer Dataset
# -----------------------------
class OptionLifetimeDataset(Dataset):
    def __init__(self, df):
        self.X = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.y = df[TARGET_COLUMNS].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# -----------------------------
# Transformer Model
# -----------------------------
class OptionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)  # batch x input_dim -> batch x hidden
        x = x.unsqueeze(1)     # batch x seq_len=1 x hidden
        x = self.transformer(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out[:,0].unsqueeze(1), out[:,1].unsqueeze(1)  # return, hold

# -----------------------------
# Load/Save
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

def append_accumulated_data(new_df: pd.DataFrame):
    TRAINING_DIR.mkdir(exist_ok=True)
    if ACCUMULATED_DATA_PATH.exists():
        old_df = pd.read_csv(ACCUMULATED_DATA_PATH)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    # drop rows where any target is NaN
    combined = combined.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    save_csv_safely(combined,ACCUMULATED_DATA_PATH,logger=logger)
    logger.logMessage(f"📈 Accumulated dataset now has {len(combined)} rows.")
    return combined
    
# -----------------------------
# Training Function (Hybrid)
# -----------------------------
def train_hybrid_model(df: pd.DataFrame):
    # Feature scaling for SGD part
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    y_return = df["predicted_return"].values
    y_hold = df["predicted_hold_days"].values

    # SGD Regressor for fast incremental updates
    sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, warm_start=True)
    sgd_model.partial_fit(X_scaled, y_return)

    # Transformer dataset
    dataset = OptionLifetimeDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    transformer_model = OptionTransformer(len(FEATURE_COLUMNS), HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    transformer_model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred_return, pred_hold = transformer_model(X_batch)
            loss = criterion(pred_return, y_batch[:,0].unsqueeze(1)) + \
                   criterion(pred_hold, y_batch[:,1].unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.logMessage(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.6f}")

    # Save hybrid
    hybrid_model = {
        "sgd": sgd_model,
        "transformer": transformer_model
    }
    save_model(hybrid_model, scaler)
    return {"status": "trained", "rows": len(df)}

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...), auto_train: bool = Form(default=False)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.dropna().reset_index(drop=True)
        accumulated_df = append_accumulated_data(df)
        result = {"status": "appended", "rows": len(df)}
        if auto_train:
            stats = train_hybrid_model(accumulated_df)
            result.update({"status": "trained", **stats})
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/train")
def train_accumulated():
    if not ACCUMULATED_DATA_PATH.exists():
        return {"status": "error", "message": "No accumulated data found."}
    df = pd.read_csv(ACCUMULATED_DATA_PATH)
    stats = train_hybrid_model(df)
    return {"status": "trained", **stats}

@app.post("/predict")
async def predict(data: dict):
    features = data.get("features", {})
    if not features:
        return {"status": "error", "message": "No features provided."}

    model_dict, scaler = load_existing_model()
    if model_dict is None:
        return {"status": "error", "message": "Model not trained yet."}

    x = np.array([features.get(f, 0) for f in FEATURE_COLUMNS]).reshape(1,-1)
    x_scaled = scaler.transform(x)

    # SGD prediction
    sgd_pred_return = model_dict["sgd"].predict(x_scaled)[0]

    # Transformer prediction
    transformer_model = model_dict["transformer"].to(DEVICE).eval()
    with torch.no_grad():
        X_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        pred_return, pred_hold = transformer_model(X_tensor)
        transformer_pred_return = pred_return.item()
        transformer_pred_hold = pred_hold.item()

    # Fusion: simple average
    final_return = (sgd_pred_return + transformer_pred_return)/2
    final_hold = transformer_pred_hold  # can also fuse hold days if needed

    signal = "BUY" if final_return > 0.05 else "SELL" if final_return < -0.05 else "HOLD"

    return {
        "status": "ok",
        "result": {
            "predicted_return": float(final_return),
            "predicted_days_to_hold": float(final_hold),
            "signal": signal
        }
    }
