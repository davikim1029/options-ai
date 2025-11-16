import torch
import joblib
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import json
import time
from fastapi.encoders import jsonable_encoder
from datetime import datetime,timezone
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from constants import LOG_EVERY_N,FEATURE_COLUMNS,BATCH_SIZE,DEVICE,MAX_SEQ_LEN_CAP,HIDDEN_DIM,TARGET_COLUMNS,LEARNING_RATE,NUM_LAYERS,MODEL_PATH,SCALER_PATH,MODEL_DIR
from utils.utils import safe_literal_eval,to_native_types
from shared_options.log.logger_singleton import getLogger

logger = getLogger()

# -----------------------------
# Target computation (Option 1: end-of-life)
# -----------------------------
def compute_end_of_life_targets(sequence):
    """
    sequence: list[dict] each with 'lastPrice' and 'timestamp'
    returns: (predicted_return: float, predicted_hold_days: int)
    target_return = (last - first) / first
    target_hold_days = ceil(delta_days) or integer days between first and last snapshot
    """
    if not isinstance(sequence, list) or len(sequence) == 0:
        return 0.0, 0
    try:
        first = sequence[0]
        last = sequence[-1]
        first_price = float(first.get("lastPrice", 0.0))
        last_price = float(last.get("lastPrice", 0.0))
        if first_price <= 1e-9:
            return 0.0, 0
        ret = (last_price - first_price) / max(first_price, 1e-9)
        # compute days between timestamps
        t0 = parse_timestamp(first.get("timestamp"))
        t1 = parse_timestamp(last.get("timestamp"))
        if t0 and t1:
            hold_days = int(max(0, (t1 - t0).days))
        else:
            # fallback to number of snapshots as proxy
            hold_days = len(sequence)
        return float(ret), int(hold_days)
    except Exception:
        return 0.0, 0

# -----------------------------
# Sequence preprocessing (pad/truncate)
# -----------------------------
def sequence_to_matrix(sequence, max_seq_len, feature_columns=FEATURE_COLUMNS):
    """
    Convert list[dict] -> np.array shape (max_seq_len, num_features).
    Truncate older snapshots (keep earliest snapshots up to max_seq_len) or keep latest?
    We'll keep the full chronological order and take the first max_seq_len snapshots (earliest->latest).
    Shorter sequences are zero-padded at the end.
    """
    num_features = len(feature_columns)
    arr = np.zeros((max_seq_len, num_features), dtype=np.float32)
    if not sequence:
        return arr
    seq_len = min(len(sequence), max_seq_len)
    for i in range(seq_len):
        snap = sequence[i]
        for j, f in enumerate(feature_columns):
            try:
                arr[i, j] = float(snap.get(f, 0.0) or 0.0)
            except Exception:
                arr[i, j] = 0.0
    return arr

# -----------------------------
# Streaming builder: group snapshots by osiKey -> yield sequence entries
# -----------------------------
def transform_for_fusion_streaming(df_snapshots, logger=None, max_seq_len_cap=MAX_SEQ_LEN_CAP):
    """
    Input: DataFrame of snapshots (rows), containing at least osiKey, timestamp, FEATURE_COLUMNS
    Output: generator yielding processed entries:
      {
        "osiKey": ...,
        "strikePrice": ...,
        "optionType": ...,
        "moneyness": ...,
        "sequence": [ {FEATURE_COLUMNS + 'timestamp'}, ... ],
        "predicted_return": target_return,
        "predicted_hold_days": target_hold_days
      }
    This function groups snapshots by osiKey and sorts by timestamp (ascending).
    It discovers an appropriate MAX_SEQ_LEN (capped).
    """
    if df_snapshots is None or df_snapshots.empty:
        return iter([])

    # Ensure timestamp ordering: convert timestamp strings to real datetimes for sorting
    if "timestamp" in df_snapshots.columns:
        # rely on pandas to_datetime (fast)
        df_snapshots["__ts"] = pd.to_datetime(df_snapshots["timestamp"], errors="coerce")
    else:
        df_snapshots["__ts"] = pd.NaT

    grouped = df_snapshots.groupby("osiKey", sort=True)

    # find max length in this batch (bounded)
    max_len_in_batch = grouped.size().max() if len(grouped) > 0 else 0
    max_seq_len = min(int(max_len_in_batch), max_seq_len_cap) if max_len_in_batch > 0 else min(64, max_seq_len_cap)
    logger and logger.logMessage(f"Determined max_seq_len={max_seq_len} for this batch")

    def generator():
        total = 0
        skipped = 0
        for osiKey, group in grouped:
            total += 1
            group = group.sort_values("__ts").reset_index(drop=True)
            # build sequence of snapshots (chronological)
            seq = []
            for _, snap in group.iterrows():
                item = {}
                # include FEATURE_COLUMNS and timestamp
                for f in FEATURE_COLUMNS:
                    item[f] = snap.get(f, 0.0)
                item["timestamp"] = snap.get("timestamp")
                seq.append(item)
            # compute targets
            pred_ret, pred_hold = compute_end_of_life_targets(seq)
            # static fields from first snapshot
            first_snap = seq[0] if seq else {}
            entry = {
                "osiKey": osiKey,
                "strikePrice": float(first_snap.get("strikePrice", 0.0)),
                "optionType": int(first_snap.get("optionType", 0) or 0),
                "moneyness": float(first_snap.get("moneyness", 0.0)),
                "predicted_return": float(pred_ret),
                "predicted_hold_days": int(pred_hold),
                "sequence": seq  # full sequence as list[dict]
            }
            if total % LOG_EVERY_N == 0:
                logger and logger.logMessage(f"transform_for_fusion_streaming: processed {total} options")
            yield entry
        logger and logger.logMessage(f"transform_for_fusion_streaming done: processed {total} options, skipped {skipped}")
    return generator()


# -----------------------------
# Dataset for training: reads CSV chunk and builds padded sequence matrices
# -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, df_chunk: pd.DataFrame, max_seq_len: int):
        """
        df_chunk is a DataFrame where each row must contain 'sequence' (JSON str or list) and target columns.
        Builds arrays: self.X shape [N, max_seq_len, F], self.y [N, 2]
        """
        rows = []
        seqs = []
        ys = []
        for _, r in df_chunk.iterrows():
            seq_raw = r.get("sequence")
            seq = None
            if isinstance(seq_raw, str):
                try:
                    seq = safe_literal_eval(seq_raw)
                    if seq is None:
                        seq = json.loads(seq_raw)
                except Exception:
                    # try json loads fallback
                    try:
                        seq = json.loads(seq_raw)
                    except Exception:
                        seq = []
            elif isinstance(seq_raw, list):
                seq = seq_raw
            else:
                seq = []
            mat = sequence_to_matrix(seq, max_seq_len)
            seqs.append(mat)
            y0 = float(r.get("predicted_return", 0.0))
            y1 = float(r.get("predicted_hold_days", 0.0))
            ys.append([y0, y1])
        if len(seqs) == 0:
            self.X = np.zeros((0, max_seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
            self.y = np.zeros((0, 2), dtype=np.float32)
        else:
            self.X = np.stack(seqs, axis=0).astype(np.float32)
            self.y = np.array(ys, dtype=np.float32)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)



# -----------------------------
# Training loop (streaming, sequence-aware)
# -----------------------------
def train_hybrid_model_streamed(csv_path: Path, batch_size=BATCH_SIZE, throttle_delay=0.0, device=DEVICE):
    """
    Streaming training:
     - SGD on first-snapshot flattened features (from CSV columns)
     - Transformer on full sequences (padded/truncated)
    """
    logger.logMessage("🚀 Starting streamed hybrid training...")

    sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, warm_start=True)
    scaler = StandardScaler()

    # determine a sensible max_seq_len by sampling a small number of rows
    # We'll scan first N rows to get distribution
    sample_max = 0
    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        for seq_raw in chunk["sequence"].head(1000).values:
            if isinstance(seq_raw, str):
                try:
                    seq_list = safe_literal_eval(seq_raw)
                    if seq_list is None:
                        seq_list = json.loads(seq_raw)
                except Exception:
                    try:
                        seq_list = json.loads(seq_raw)
                    except Exception:
                        seq_list = []
            elif isinstance(seq_raw, list):
                seq_list = seq_raw
            else:
                seq_list = []
            sample_max = max(sample_max, len(seq_list))
        break
    max_seq_len = min(max(sample_max, 8), MAX_SEQ_LEN_CAP)
    logger.logMessage(f"Using max_seq_len={max_seq_len} for training")

    transformer_model = OptionTransformer(len(FEATURE_COLUMNS), HIDDEN_DIM, NUM_LAYERS, max_seq_len).to(device)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    transformer_model.train()

    total_rows = 0
    first_chunk = True

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=10_000)):
        # ensure required columns
        for c in FEATURE_COLUMNS + TARGET_COLUMNS + ["sequence"]:
            if c not in chunk.columns:
                chunk[c] = 0.0 if c in TARGET_COLUMNS else ""
        chunk = chunk.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)
        if chunk.empty:
            continue

        # SGD training on first-snapshot features (we already saved first snapshot's features in top-level columns)
        X_first = chunk[FEATURE_COLUMNS].values.astype(np.float32)
        y_first = chunk[TARGET_COLUMNS].values.astype(np.float32)
        if first_chunk:
            X_scaled = scaler.fit_transform(X_first)
            first_chunk = False
        else:
            X_scaled = scaler.transform(X_first)
        sgd_model.partial_fit(X_scaled, y_first[:, 0])

        # Transformer training: build SequenceDataset for this chunk
        seq_ds = SequenceDataset(chunk, max_seq_len)
        if len(seq_ds) == 0:
            total_rows += len(chunk)
            continue
        dataloader = DataLoader(seq_ds, batch_size=batch_size, shuffle=True)

        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred_return, pred_hold = transformer_model(X_batch)
            # pred_return, pred_hold shapes are [B], y_batch columns are [B]
            loss = criterion(pred_return, y_batch[:, 0]) + criterion(pred_hold, y_batch[:, 1])
            loss.backward()
            optimizer.step()
            if throttle_delay > 0:
                time.sleep(throttle_delay)

            if batch_idx % 10 == 0:
                logger.logMessage(f"Chunk {chunk_idx} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")

        total_rows += len(chunk)
        logger.logMessage(f"✅ Finished chunk {chunk_idx} | Total rows processed: {total_rows}")

    # Save model & scaler
    hybrid_model = {"sgd": sgd_model, "transformer": transformer_model}
    save_model(hybrid_model, scaler)
    logger.logMessage(f"🎯 Streamed hybrid training complete: {total_rows} rows processed")

    return jsonable_encoder(to_native_types({"status": "trained", "rows": int(total_rows)}))


# -----------------------------
# Save / Load helpers
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
# Utility helpers (streaming safe)
# -----------------------------
def now_iso():
    return datetime.now(tz=timezone.utc).isoformat()

def parse_timestamp(ts):
    """Parse timestamp strings from DB. Accepts ISO or naive strings."""
    if ts is None:
        return None
    try:
        # if already datetime
        if isinstance(ts, datetime):
            return ts
        # try iso first
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


# -----------------------------
# Model classes (sequence-aware)
# -----------------------------
class OptionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x, mask=None):
        """
        x: [B, L, F]
        mask: optional boolean mask [B, L] where True indicates valid tokens (not required here because we zero-pad)
        returns: (pred_return [B], pred_hold [B])
        """
        emb = self.embedding(x)           # [B, L, H]
        out = self.transformer(emb)       # [B, L, H]
        pooled = out.mean(dim=1)          # [B, H]
        out = self.fc_out(pooled)         # [B, 2]
        return out[:, 0], out[:, 1]
