import os
import io
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Fusion Transformer AI Model Service")

# ------------------------------
# Directories & Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"
MODEL_DIR = BASE_DIR / "models" / "versions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "current_model.pth"
ACCUMULATED_DATA_PATH = TRAINING_DIR / "accumulated_training.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

# ------------------------------
# Transformer Dataset
# ------------------------------
class OptionLifetimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.groups = []
        feature_cols = [
            "optionType", "strikePrice", "lastPrice", "bid", "ask", "bidSize", "askSize",
            "volume", "openInterest", "nearPrice", "inTheMoney", "delta", "gamma", "theta",
            "vega", "rho", "iv", "spread", "midPrice", "moneyness", "daysToExpiration"
        ]
        target_cols = ["return", "hold_days"]

        for osiKey, group in df.groupby("osiKey"):
            X = torch.tensor(group[feature_cols].values, dtype=torch.float32)
            y = torch.tensor(group[target_cols].values, dtype=torch.float32)
            self.groups.append((X, y))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]

# ------------------------------
# Fusion Transformer Model
# ------------------------------
class FusionTransformer(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.hold_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: seq_len x batch x input_dim
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=0)  # aggregate over time steps
        return self.return_head(x), self.hold_head(x)

# ------------------------------
# Helpers
# ------------------------------
def load_model():
    model = FusionTransformer().to(DEVICE)
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ Loaded existing transformer model.")
    return model

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Saved model -> {MODEL_PATH}")

def append_accumulated_data(new_df: pd.DataFrame):
    TRAINING_DIR.mkdir(exist_ok=True)
    if ACCUMULATED_DATA_PATH.exists():
        old_df = pd.read_csv(ACCUMULATED_DATA_PATH)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(ACCUMULATED_DATA_PATH, index=False)
    print(f"📈 Accumulated dataset now has {len(combined)} rows.")
    return combined

def train_model(df: pd.DataFrame):
    dataset = OptionLifetimeDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = load_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            X = X.transpose(0, 1)  # seq_len x batch x input_dim
            pred_return, pred_hold = model(X)
            loss = criterion(pred_return, y[:,0].unsqueeze(1)) + criterion(pred_hold, y[:,1].unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.6f}")

    save_model(model)
    return {"status": "trained", "rows": len(df)}

# ------------------------------
# API Endpoints
# ------------------------------
@app.post("/train/upload")
async def upload_training(file: UploadFile = File(...), auto_train: bool = Form(default=False)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.dropna().reset_index(drop=True)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    accumulated_df = append_accumulated_data(df)
    result = {"status": "appended", "rows": len(df)}

    if auto_train:
        train_result = train_model(accumulated_df)
        result.update(train_result)

    return result

@app.post("/train")
def train_accumulated():
    if not ACCUMULATED_DATA_PATH.exists():
        return {"status": "error", "message": "No accumulated data found."}
    df = pd.read_csv(ACCUMULATED_DATA_PATH)
    return train_model(df)

@app.post("/predict")
async def predict(data: dict):
    features = data.get("features", {})
    if not features:
        return {"status": "error", "message": "No features provided."}

    model = load_model().eval()
    feature_order = [
        "optionType", "strikePrice", "lastPrice", "bid", "ask", "bidSize", "askSize",
        "volume", "openInterest", "nearPrice", "inTheMoney", "delta", "gamma", "theta",
        "vega", "rho", "iv", "spread", "midPrice", "moneyness", "daysToExpiration"
    ]
    x = torch.tensor([[features.get(f, 0) for f in feature_order]], dtype=torch.float32).to(DEVICE)
    x = x.transpose(0,1)  # seq_len x batch x input_dim
    with torch.no_grad():
        pred_return, pred_hold = model(x)
    pred_return_val = float(pred_return.item())
    pred_hold_val = float(pred_hold.item())
    signal = "BUY" if pred_return_val > 0.05 else "SELL" if pred_return_val < -0.05 else "HOLD"

    return {
        "status": "ok",
        "result": {
            "predicted_return": pred_return_val,
            "predicted_days_to_hold": pred_hold_val,
            "signal": signal
        }
    }
