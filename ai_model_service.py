# ai_server.py
import io
import joblib
import time
import tempfile
import json
import math
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
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
MAX_SEQ_LEN = 50

# -----------------------------
# Utility helpers
# -----------------------------
def to_native_types(obj):
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

# -----------------------------
# Dataset / Model
# -----------------------------
class OptionLifetimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.samples = []
        for _, row in df.iterrows():
            seq = row.get("sequence")
            if isinstance(seq, str):
                seq = safe_literal_eval(seq) or []
            elif not isinstance(seq, list):
                seq = []
            seq_len = min(len(seq), self.max_seq_len)
            seq_features = np.zeros((self.max_seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
            for i in range(seq_len):
                for j, f in enumerate(FEATURE_COLUMNS):
                    seq_features[i, j] = float(seq[i].get(f, 0.0))
            y = np.array([
                float(row.get("predicted_return", 0.0)),
                float(row.get("predicted_hold_days", 0))
            ], dtype=np.float32)
            self.samples.append((seq_features, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class OptionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: [B, L, F]
        x = self.embedding(x)               # [B, L, H]
        x = x.transpose(0, 1)               # [L, B, H]
        x = self.transformer(x)             # [L, B, H]
        x = x.mean(dim=0)                   # [B, H]
        out = self.fc_out(x)                # [B, 2]
        return out[:, 0], out[:, 1]

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
# Label Generation
# -----------------------------
def compute_max_profit_after_valley(sequence, min_expected_gain=0.01, min_risk_reward=0.1, smoothing_window=3, late_game_bump_threshold=0.02):
    if not sequence:
        return 0.0, 0
    prices = [float(x.get("lastPrice", 0.0)) for x in sequence]
    n = len(prices)
    smoothed = [max(prices[i:min(n,i+smoothing_window)]) for i in range(n)]
    entry_price = prices[0]
    max_profit, hold_days_at_max = -float("inf"), 0
    in_valley, current_valley, min_future_price = False, entry_price, entry_price
    for j in range(n):
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
    if max_profit == -float("inf"):
        future_profits = [(p - entry_price)/max(entry_price,1e-9) for p in prices]
        max_profit = max(future_profits)
        hold_days_at_max = future_profits.index(max_profit)
    hold_days_at_max = min(hold_days_at_max, n-1)
    potential_loss = max(entry_price - min_future_price, 1e-9)
    risk_reward_ratio = max_profit/potential_loss if potential_loss>0 else float("inf")
    predicted_return = float(max_profit) if (max_profit>=min_expected_gain and risk_reward_ratio>=min_risk_reward) else 0.0
    predicted_hold_days = int(hold_days_at_max) if predicted_return>0 else 0
    remaining_days = n
    if remaining_days>0 and predicted_hold_days>0 and (predicted_hold_days/remaining_days)>0.8 and predicted_return<late_game_bump_threshold:
        predicted_return=0.0
        predicted_hold_days=0
    return predicted_return, predicted_hold_days

def flatten_entry_for_training(entry, max_seq_len=MAX_SEQ_LEN):
    seq = entry.get("sequence")
    if isinstance(seq,str):
        seq = safe_literal_eval(seq) or []
    elif not isinstance(seq,list):
        seq=[]
    seq_len=min(len(seq),max_seq_len)
    seq_features=[]
    for i in range(seq_len):
        snapshot = {f: float(seq[i].get(f,0.0)) for f in FEATURE_COLUMNS}
        seq_features.append(snapshot)
    predicted_return, predicted_hold_days = compute_max_profit_after_valley(seq)
    first_snapshot = seq_features[0] if seq_features else {f:0.0 for f in FEATURE_COLUMNS}
    flat = first_snapshot.copy()
    flat["predicted_return"]=float(predicted_return)
    flat["predicted_hold_days"]=int(predicted_hold_days)
    flat["sequence"]=seq_features
    return flat

# -----------------------------
# CSV helpers
# -----------------------------
def save_rows_to_csv_stream(rows_iterable, out_path: Path, chunk_size: int=5000):
    TRAINING_DIR.mkdir(exist_ok=True)
    tmp=out_path.with_suffix(".tmp")
    header_written=out_path.exists()
    buffer=[]
    written=0
    for row in rows_iterable:
        row = to_native_types(row)
        buffer.append(row)
        if len(buffer)>=chunk_size:
            df=pd.DataFrame(buffer)
            df.to_csv(tmp,mode="a",index=False,header=(not tmp.exists() and not header_written))
            written+=len(buffer)
            buffer.clear()
    if buffer:
        df=pd.DataFrame(buffer)
        df.to_csv(tmp,mode="a",index=False,header=(not tmp.exists() and not header_written))
        written+=len(buffer)
    if not out_path.exists() and tmp.exists():
        tmp.replace(out_path)
    elif tmp.exists():
        with open(out_path,"a") as fa, open(tmp,"r") as ft:
            first=True
            for line in ft:
                if first: first=False; continue
                fa.write(line)
        tmp.unlink(missing_ok=True)
    return written

def append_accumulated_data_append_only(new_df: pd.DataFrame):
    TRAINING_DIR.mkdir(exist_ok=True)
    for col in new_df.select_dtypes(include=[np.integer,np.floating]).columns:
        if new_df[col].dtype.kind in ("i","u"):
            new_df[col]=new_df[col].astype(np.int64)
        else:
            new_df[col]=new_df[col].astype(np.float64)
    mode="a" if ACCUMULATED_DATA_PATH.exists() else "w"
    header = not ACCUMULATED_DATA_PATH.exists()
    new_df.to_csv(ACCUMULATED_DATA_PATH, mode=mode, index=False, header=header)
    total=sum(1 for _ in pd.read_csv(ACCUMULATED_DATA_PATH, chunksize=100_000))
    logger.logMessage(f"📈 Accumulated dataset now has {total} rows.")
    return ACCUMULATED_DATA_PATH,len(new_df)

# -----------------------------
# Streaming hybrid training
# -----------------------------
def train_hybrid_model_streamed(csv_path: Path, batch_size=BATCH_SIZE, throttle_delay=0.05, device=DEVICE):
    logger.logMessage("Starting streamed hybrid training...")
    sgd_model=SGDRegressor(max_iter=1000,tol=1e-3,warm_start=True)
    scaler=StandardScaler()
    transformer_model=OptionTransformer(len(FEATURE_COLUMNS),HIDDEN_DIM,NUM_LAYERS).to(device)
    optimizer=torch.optim.Adam(transformer_model.parameters(),lr=LEARNING_RATE)
    criterion=nn.MSELoss()
    transformer_model.train()
    total_rows=0
    first_chunk=True
    for chunk in pd.read_csv(csv_path,chunksize=50_000):
        for c in FEATURE_COLUMNS+TARGET_COLUMNS:
            if c not in chunk.columns: chunk[c]=0.0
        chunk=chunk.dropna(subset=FEATURE_COLUMNS+TARGET_COLUMNS).reset_index(drop=True)
        if chunk.empty: continue
        # SGD training
        X_first=chunk[FEATURE_COLUMNS].values.astype(np.float32)
        y_first=chunk[TARGET_COLUMNS].values.astype(np.float32)
        if first_chunk:
            X_scaled=scaler.fit_transform(X_first)
            first_chunk=False
        else:
            X_scaled=scaler.transform(X_first)
        sgd_model.partial_fit(X_scaled,y_first[:,0])
        # Transformer training
        sequence_rows=[]
        for _, row in chunk.iterrows():
            seq=row.get("sequence")
            if isinstance(seq,str):
                seq=safe_literal_eval(seq) or []
            if not isinstance(seq,list) or len(seq)==0: continue
            seq_array=np.array([[float(s.get(f,0.0)) for f in FEATURE_COLUMNS] for s in seq],dtype=np.float32)
            target_array=np.array([[float(s.get("predicted_return",0.0)),float(s.get("predicted_hold_days",0))] for s in seq],dtype=np.float32)
            sequence_rows.append((seq_array,target_array))
        if not sequence_rows: continue
        class SequenceDataset(Dataset):
            def __init__(self,data): self.data=data
            def __len__(self): return len(self.data)
            def __getitem__(self,idx): return self.data[idx]
        dataset=SequenceDataset(sequence_rows)
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=lambda x:x)
        for batch in dataloader:
            optimizer.zero_grad()
            seqs=[torch.tensor(b[0],dtype=torch.float32) for b in batch]
            targets=[torch.tensor(b[1],dtype=torch.float32) for b in batch]
            seq_lens=[s.shape[0] for s in seqs]
            max_len=max(seq_lens)
            padded_seqs=torch.stack([torch.cat([s,torch.zeros(max_len-s.shape[0],s.shape[1])],dim=0) for s in seqs]).to(device)
            padded_targets=torch.stack([torch.cat([t,torch.zeros(max_len-t.shape[0],t.shape[1])],dim=0) for t in targets]).to(device)
            pred_return,pred_hold=transformer_model(padded_seqs)
            loss=criterion(pred_return,padded_targets[:,0]) + criterion(pred_hold,padded_targets[:,1])
            loss.backward()
            optimizer.step()
            if throttle_delay>0: time.sleep(throttle_delay)
        total_rows+=len(chunk)
    hybrid_model={"sgd":sgd_model,"transformer":transformer_model}
    save_model(hybrid_model,scaler)
    logger.logMessage(f"📊 Streamed hybrid training complete: {total_rows} rows processed")
    return jsonable_encoder(to_native_types({"status":"trained","rows":int(total_rows)}))

# -----------------------------
# Upload endpoint
# -----------------------------
@app.post("/train/upload")
async def upload_training_data(file: UploadFile=File(...), auto_train: bool=Form(default=False)):
    try:
        TRAINING_DIR.mkdir(exist_ok=True)
        tmp_path=TRAINING_DIR/f"_upload_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(tmp_path,"wb") as fw:
            while chunk:=await file.read(1024*1024):
                fw.write(chunk)
        is_json_like=False
        with open(tmp_path,"rb") as fr:
            start=fr.read(1024).lstrip()
            if start.startswith(b"{") or start.startswith(b"["): is_json_like=True
        total_appended=0
        if is_json_like:
            with open(tmp_path,"r",encoding="utf-8") as r:
                data=json.load(r)
            def generator(): 
                for entry in data:
                    try: yield flatten_entry_for_training(entry)
                    except Exception as e: logger.logMessage(f"⚠️ Skipping malformed entry: {e}")
            tmp_csv=TRAINING_DIR/f"_upload_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            written=save_rows_to_csv_stream(generator(),tmp_csv,chunk_size=5000)
            if written:
                df_chunk=pd.read_csv(tmp_csv)
                path,cnt=append_accumulated_data_append_only(df_chunk)
                total_appended+=cnt
                tmp_csv.unlink(missing_ok=True)
        else:
            for chunk in pd.read_csv(tmp_path,chunksize=50_000):
                rows=[]
                if "sequence" in chunk.columns:
                    for _,r in chunk.iterrows():
                        seq_val=r.get("sequence")
                        seq=None
                        if isinstance(seq_val,str): seq=safe_literal_eval(seq_val)
                        elif isinstance(seq_val,list): seq=seq_val
                        else: seq=[]
                        entry={"osiKey":r.get("osiKey"),"strikePrice":r.get("strikePrice"),
                               "optionType":r.get("optionType"),"moneyness":r.get("moneyness"),
                               "sequence":seq}
                        try: rows.append(flatten_entry_for_training(entry))
                        except Exception as e: logger.logMessage(f"⚠️ Skipping row: {e}")
                    if rows:
                        df_chunk=pd.DataFrame(rows)
                        append_accumulated_data_append_only(df_chunk)
                        total_appended+=len(df_chunk)
                else:
                    append_accumulated_data_append_only(chunk)
                    total_appended+=len(chunk)
        tmp_path.unlink(missing_ok=True)
        result={"status":"appended","rows":int(total_appended)}
        if auto_train:
            train_result=train_hybrid_model_streamed(ACCUMULATED_DATA_PATH,batch_size=BATCH_SIZE,throttle_delay=0.05,device=DEVICE)
            return train_result
        return jsonable_encoder(to_native_types(result))
    except Exception as e:
        logger.logMessage(f"Upload error: {e}")
        return jsonable_encoder(to_native_types({"status":"error","message":str(e)}))

# -----------------------------
# Train endpoint
# -----------------------------
@app.post("/train")
def train_accumulated():
    try:
        if not ACCUMULATED_DATA_PATH.exists():
            return jsonable_encoder(to_native_types({"status":"error","message":"No accumulated data found."}))
        result=train_hybrid_model_streamed(ACCUMULATED_DATA_PATH,batch_size=BATCH_SIZE,throttle_delay=0.05,device=DEVICE)
        return result
    except Exception as e:
        logger.logMessage(f"Training error: {e}")
        return jsonable_encoder(to_native_types({"status":"error","message":str(e)}))


# -----------------------------
# Predict endpoint (refactored)
# -----------------------------
@app.post("/predict")
async def predict(data: dict):
    """
    Predict an option's expected return and hold days using:
     - 'features' (first snapshot) OR
     - 'sequence' (full lifetime)
     
    Returns JSON with predicted_return, predicted_days_to_hold, and signal.
    """
    try:
        features = data.get("features", None)
        sequence = data.get("sequence", None)

        result = predict_option(features=features, sequence=sequence)
        return jsonable_encoder(to_native_types(result))

    except Exception as e:
        logger.logMessage(f"Predict error: {e}")
        return jsonable_encoder(to_native_types({"status": "error", "message": str(e)}))


# -----------------------------
# Lightweight inference helper
# -----------------------------
def predict_option(features: dict = None, sequence: list = None):
    """
    Predict an option's expected return and hold days.
    Automatically handles variable-length sequences and padding.
    
    Args:
        features: dict of snapshot features (first snapshot if sequence missing)
        sequence: list of snapshots (each snapshot is a dict of FEATURE_COLUMNS)
    
    Returns:
        dict with predicted_return, predicted_days_to_hold, and signal
    """
    model_dict, scaler = load_existing_model()
    if model_dict is None:
        return {"status": "error", "message": "Model not trained yet."}

    # Prepare first snapshot if no sequence provided
    if sequence is None or len(sequence) == 0:
        if features is None:
            return {"status": "error", "message": "No features or sequence provided."}
        sequence = [features]
    else:
        if features is None:
            features = sequence[0]

    # SGD prediction (first snapshot)
    x_flat = np.array([features.get(f, 0) for f in FEATURE_COLUMNS]).reshape(1, -1)
    x_scaled = scaler.transform(x_flat)
    sgd_pred_return = model_dict["sgd"].predict(x_scaled)[0]

    # Transformer prediction (full sequence)
    seq_len = min(len(sequence), MAX_SEQ_LEN)
    seq_features = np.zeros((MAX_SEQ_LEN, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i in range(seq_len):
        seq_features[i] = [sequence[i].get(f, 0) for f in FEATURE_COLUMNS]
    X_tensor = torch.tensor(seq_features[:seq_len], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    transformer_model = model_dict["transformer"].to(DEVICE).eval()
    with torch.no_grad():
        transformer_pred_return, transformer_pred_hold = transformer_model(X_tensor)
        transformer_pred_return = float(transformer_pred_return.item())
        transformer_pred_hold = float(transformer_pred_hold.item())

    # Final combined prediction
    final_return = float((sgd_pred_return + transformer_pred_return) / 2.0)
    final_hold = transformer_pred_hold

    # Simple signal logic
    signal = "BUY" if final_return > 0.05 else "SELL" if final_return < -0.05 else "HOLD"

    return {
        "status": "ok",
        "predicted_return": final_return,
        "predicted_days_to_hold": final_hold,
        "signal": signal
    }
