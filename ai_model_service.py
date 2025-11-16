# ai_server.py  (FULL production-ready rewrite - drop-in replacement)
import joblib
import uuid
import time
import json
import math
import sqlite3
import ijson
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, Body,APIRouter,WebSocket
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from logger.logger_singleton import getLogger
from utils.utils import safe_literal_eval, to_native_types
from pydantic import BaseModel
from evaluation import evaluate_model
from backtester_api import router as backtest_router
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


logger = getLogger()
app = FastAPI(title="Hybrid AI Model Service (Seq Transformer)")
app.include_router(backtest_router)
app.add_middleware(RequestIDMiddleware)


# -----------------------------
# Paths & Constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"
MODEL_DIR = BASE_DIR / "models" / "versions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "current_model.pkl"
SCALER_PATH = MODEL_DIR / "current_scaler.pkl"
ACCUMULATED_DATA_PATH = TRAINING_DIR / "accumulated_training.csv"  # stores flattened first-snapshot + sequence column (json)

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

# Maximum sequence length to use (cap). We'll auto-discover from a chunk but never exceed this.
MAX_SEQ_LEN_CAP = 250

# Logging frequency
LOG_EVERY_N = 10_000

# -----------------------------
# Utility helpers (streaming safe)
# -----------------------------
def ensure_training_dir():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

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
# CSV helpers (streaming safe)
# -----------------------------
def save_rows_to_csv_stream(rows_iterable, out_path: Path, chunk_size: int = 5000):
    """
    Write rows (dicts) into CSV incrementally. rows_iterable yields dict rows.
    Stores 'sequence' column as JSON string.
    """
    ensure_training_dir()
    tmp = out_path.with_suffix(".tmp")
    header_written = out_path.exists()
    buffer = []
    written = 0
    for row in rows_iterable:
        # convert sequence to JSON string & ensure native types
        r = dict(row)
        seq = r.get("sequence", [])
        r["sequence"] = json.dumps(to_native_types(seq), ensure_ascii=False)
        # ensure numeric targets are python types
        for k in TARGET_COLUMNS:
            if k in r:
                if isinstance(r[k], (np.integer,)):
                    r[k] = int(r[k])
                if isinstance(r[k], (np.floating,)):
                    r[k] = float(r[k])
        buffer.append(r)
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
    if not out_path.exists() and tmp.exists():
        tmp.replace(out_path)
    else:
        if tmp.exists():
            with open(out_path, "a") as fa, open(tmp, "r") as ft:
                first_line = True
                for line in ft:
                    if first_line:
                        first_line = False
                        continue
                    fa.write(line)
            tmp.unlink(missing_ok=True)
    return written

def append_accumulated_data_append_only(new_df: pd.DataFrame):
    """Append chunk to ACCUMULATED_DATA_PATH in append-only manner (stream-safe)."""
    ensure_training_dir()
    mode = "a" if ACCUMULATED_DATA_PATH.exists() else "w"
    header = not ACCUMULATED_DATA_PATH.exists()
    # cast numeric numpy types to python-native before saving
    new_df = new_df.copy()
    for col in new_df.select_dtypes(include=[np.integer, np.floating]).columns:
        if new_df[col].dtype.kind in ("i", "u"):
            new_df[col] = new_df[col].astype(np.int64)
        else:
            new_df[col] = new_df[col].astype(np.float64)
    new_df.to_csv(ACCUMULATED_DATA_PATH, mode=mode, index=False, header=header)
    total = sum(1 for _ in pd.read_csv(ACCUMULATED_DATA_PATH, chunksize=100_000))
    logger.logMessage(f"📈 Accumulated dataset now has {total} rows.")
    return ACCUMULATED_DATA_PATH, len(new_df)

# -----------------------------
# Upload endpoint (streaming + ijson)
# -----------------------------

router = APIRouter()
logger = getLogger()

# -------------------------
# In-memory store for progress
# -------------------------
upload_progress = {}  # key: upload_id, value: dict with processed counts etc.


@router.websocket("/ws/progress/{upload_id}")
async def websocket_progress(ws: WebSocket, upload_id: str):
    await ws.accept()
    try:
        while True:
            if upload_id in upload_progress:
                progress_data = upload_progress[upload_id].copy()
                await ws.send_json(progress_data)
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.logMessage(f"WebSocket closed for {upload_id}: {e}")


def log_progress_ws(upload_id: str, processed: int, message="Processed", total: int = None):
    """Update progress dictionary for WebSocket clients."""
    start_time = upload_progress[upload_id].get("start_time", time.time())
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / rate if total and rate > 0 else None

    upload_progress[upload_id].update({
        "processed": processed,
        "rate_per_s": rate,
        "message": message,
        "eta": eta,
        "timestamp": time.time()
    })
    # Also log to standard logger occasionally
    if processed % LOG_EVERY_N == 0:
        eta_str = f", ETA: {eta:.1f}s" if eta else ""
        logger.logMessage(f"{message}: {processed}, Rate: {rate:.1f}/s{eta_str}")


@router.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...), auto_train: bool = Form(default=True)):
    """
    Streaming-safe upload with live WebSocket progress reporting.
    """
    upload_id = str(uuid.uuid4())
    upload_progress[upload_id] = {"processed": 0, "rate_per_s": 0, "message": "Starting", "start_time": time.time()}

    tmp_upload = TRAINING_DIR / "_upload_tmp_training.json"

    try:
        ensure_training_dir()

        # -------------------
        # Stream upload to disk
        # -------------------
        uploaded_bytes = 0
        total_bytes = file.headers.get("content-length")
        with open(tmp_upload, "wb") as fw:
            while chunk := await file.read(1024 * 1024):
                fw.write(chunk)
                uploaded_bytes += len(chunk)
                pct = (uploaded_bytes / int(total_bytes) * 100) if total_bytes else None
                upload_progress[upload_id].update({"message": f"Uploading: {pct:.1f}%" if pct else "Uploading..."})

        logger.logMessage(f"Upload saved to disk: {tmp_upload}")

        # -------------------
        # Peek first item to detect type
        # -------------------
        is_per_option = False
        with open(tmp_upload, "rb") as fr:
            try:
                first_item = next(ijson.items(fr, "item"))
                if first_item and isinstance(first_item, dict) and "sequence" in first_item:
                    is_per_option = True
            except StopIteration:
                return jsonable_encoder({"status": "error", "message": "Uploaded file is empty"})

        # -------------------
        # Generators with WebSocket progress
        # -------------------
        def gen_per_option():
            processed = 0
            skipped = 0
            for item in ijson.items(open(tmp_upload, "rb"), "item"):
                try:
                    seq = item.get("sequence")
                    if isinstance(seq, str):
                        seq = safe_literal_eval(seq) or []
                    pred_ret, pred_hold = compute_end_of_life_targets(seq)
                    first_snap = seq[0] if seq else {}
                    row = {
                        "osiKey": item.get("osiKey") or first_snap.get("osiKey"),
                        "strikePrice": float(item.get("strikePrice", first_snap.get("strikePrice", 0.0) or 0.0)),
                        "optionType": int(item.get("optionType", first_snap.get("optionType", 0) or 0)),
                        "moneyness": float(item.get("moneyness", first_snap.get("moneyness", 0.0) or 0.0)),
                        "predicted_return": float(pred_ret),
                        "predicted_hold_days": int(pred_hold),
                        "sequence": seq
                    }
                    processed += 1
                    if processed % LOG_EVERY_N == 0:
                        log_progress_ws(upload_id, processed, "Per-option entries processed")
                    yield row
                except Exception as e:
                    skipped += 1
                    logger.logMessage(f"⚠️ Skipping malformed per-option upload item: {e}")
                    continue

        def gen_snapshot():
            tmp_sqlite = TRAINING_DIR / f"_upload_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            conn = sqlite3.connect(tmp_sqlite)
            cur = conn.cursor()
            cols = ["osiKey TEXT", "timestamp TEXT"] + [f"{c} TEXT" for c in FEATURE_COLUMNS]
            cur.execute(f"CREATE TABLE snaps ({', '.join(cols)})")
            conn.commit()

            inserted = 0
            skipped = 0
            for item in ijson.items(open(tmp_upload, "rb"), "item"):
                try:
                    osi = item.get("osiKey")
                    ts = item.get("timestamp")
                    vals = [str(item.get(c, "")) for c in FEATURE_COLUMNS]
                    cur.execute(f"INSERT INTO snaps VALUES ({', '.join(['?'] * (2 + len(FEATURE_COLUMNS)))})",
                                [osi, ts] + vals)
                    inserted += 1
                    if inserted % LOG_EVERY_N == 0:
                        conn.commit()
                        log_progress_ws(upload_id, inserted, "Snapshot rows inserted")
                except Exception as e:
                    skipped += 1
                    logger.logMessage(f"⚠️ Skipping malformed snapshot upload item: {e}")
            conn.commit()

            buffer_by_osi = {}
            processed = 0
            for chunk_df in pd.read_sql_query("SELECT * FROM snaps ORDER BY osiKey, timestamp", conn, chunksize=50_000):
                for _, r in chunk_df.iterrows():
                    osi = r["osiKey"]
                    if osi is None:
                        continue
                    snap = {"timestamp": r["timestamp"]}
                    for c in FEATURE_COLUMNS:
                        raw = r[c]
                        try:
                            snap[c] = float(raw) if raw not in (None, "", "None") else 0.0
                        except Exception:
                            snap[c] = 0.0
                    buffer_by_osi.setdefault(osi, []).append(snap)
                    if len(buffer_by_osi) > 5000:
                        for k in list(buffer_by_osi.keys())[:1000]:
                            seq = buffer_by_osi.pop(k)
                            pr, ph = compute_end_of_life_targets(seq)
                            first = seq[0] if seq else {}
                            row = {
                                "osiKey": k,
                                "strikePrice": float(first.get("strikePrice", 0.0)),
                                "optionType": int(first.get("optionType", 0) or 0),
                                "moneyness": float(first.get("moneyness", 0.0) or 0.0),
                                "predicted_return": float(pr),
                                "predicted_hold_days": int(ph),
                                "sequence": seq
                            }
                            processed += 1
                            if processed % LOG_EVERY_N == 0:
                                log_progress_ws(upload_id, processed, "Grouped snapshot entries processed")
                            yield row
            # flush remaining
            for k, seq in buffer_by_osi.items():
                pr, ph = compute_end_of_life_targets(seq)
                first = seq[0] if seq else {}
                row = {
                    "osiKey": k,
                    "strikePrice": float(first.get("strikePrice", 0.0)),
                    "optionType": int(first.get("optionType", 0) or 0),
                    "moneyness": float(first.get("moneyness", 0.0) or 0.0),
                    "predicted_return": float(pr),
                    "predicted_hold_days": int(ph),
                    "sequence": seq
                }
                processed += 1
                if processed % LOG_EVERY_N == 0:
                    log_progress_ws(upload_id, processed, "Grouped snapshot entries processed")
                yield row
            conn.close()
            tmp_sqlite.unlink(missing_ok=True)

        # -------------------
        # Save to CSV streaming
        # -------------------
        if is_per_option:
            save_rows_to_csv_stream(gen_per_option(), ACCUMULATED_DATA_PATH, chunk_size=5000)
        else:
            save_rows_to_csv_stream(gen_snapshot(), ACCUMULATED_DATA_PATH, chunk_size=5000)
        logger.logMessage("✅ Upload processing complete and appended to accumulated CSV")

        # -------------------
        # Optional auto_train
        # -------------------
        if auto_train:
            res = train_hybrid_model_streamed(ACCUMULATED_DATA_PATH, batch_size=BATCH_SIZE, throttle_delay=0.0, device=DEVICE)
            return res

        return jsonable_encoder({"status": "appended", "upload_id": upload_id})

    except Exception as e:
        logger.logMessage(f"❌ Upload error [Server Side]: {e}")
        return jsonable_encoder({"status": "error", "message": str(e)})

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
