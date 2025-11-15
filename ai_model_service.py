# ai_server.py  (FULL production-ready rewrite - drop-in replacement)
import io
import joblib
import time
import json
import math
import ast
import sqlite3
import ijson  # streaming JSON parser
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from logger.logger_singleton import getLogger
from utils.utils import safe_literal_eval, to_native_types
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File
from evaluation import evaluate_model
from backtester import BACKTEST_STATUS_PATH, run_backtest_streaming
from dataloader import load_lifetime_dataset,load_accumulated_training_csvs
from fastapi import APIRouter

logger = getLogger()
app = FastAPI(title="Hybrid AI Model Service (Seq Transformer)")

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
@app.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...), auto_train: bool = Form(default=False)):
    """
    Stream a potentially huge JSON file to disk, stream-parse with ijson,
    group into sequences, compute targets, and append to accumulated CSV in chunks.
    """
    try:
        ensure_training_dir()
        tmp_upload = TRAINING_DIR / f"_upload_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # 1) Save upload to disk (streamed)
        with open(tmp_upload, "wb") as fw:
            while chunk := await file.read(1024 * 1024):
                fw.write(chunk)

        # 2) Stream-parse and group by osiKey using pandas grouping on the fly:
        # We'll read file with ijson.items and accumulate per-osiKey rows into an in-memory buffer sized by unique osiKey,
        # but to avoid huge memory we flush once buffer size exceeds a threshold by building processed entries and writing CSV chunk.
        #
        # Strategy:
        #  - Make a generator that yields raw entries (the uploaded JSON is expected to be list of snapshot rows OR list of per-option objects).
        #  - If snapshots (flat rows with osiKey), we group by osiKey using an on-disk temporary SQLite table to avoid large memory usage.
        #
        # Detect whether uploaded file contains per-option entries (each item has 'sequence') or snapshot-level rows
        is_per_option = False
        with open(tmp_upload, "rb") as fr:
            parser = ijson.parse(fr)
            # find the first meaningful item shape quickly
            first_item = None
            for prefix, event, value in parser:
                if prefix == "" and event == "start_array":
                    continue
                if event == "start_map":
                    # read the map into a dict (use ijson.items to avoid reading entire file)
                    first_item = True  # we know there's at least an object; we'll inspect more below
                    break
            # fall back: assume snapshots if not clearly per-option
            fr.seek(0)

        # We will use a small SQLite temp table to group snapshots by osiKey if file is snapshots list.
        # If uploaded items already have 'sequence' key, we directly flatten them to CSV using flatten_entry_for_training.
        # We'll peek one item to see if it contains 'sequence'.
        with open(tmp_upload, "rb") as fr:
            items = ijson.items(fr, "item")
            # peek one safely
            try:
                first = next(items)
            except StopIteration:
                first = None
            # check shape
            if first and isinstance(first, dict) and "sequence" in first:
                is_per_option = True
            # rewind by reopening iterator
        # reopen for real processing
        with open(tmp_upload, "rb") as fr:
            items = ijson.items(fr, "item")
            # If it's per-option entries (sequence already assembled), process directly
            if is_per_option:
                logger.logMessage("Upload detected per-option entries (contains 'sequence'). Streaming processing...")
                def gen_per_option():
                    processed = 0
                    skipped = 0
                    for item in ijson.items(open(tmp_upload, "rb"), "item"):
                        try:
                            # each item already has sequence => compute targets and yield flat row
                            # but items may already have predicted targets; we recompute to be consistent
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
                                logger.logMessage(f"Uploaded per-option entries processed: {processed}")
                            yield row
                        except Exception as e:
                            skipped += 1
                            logger.logMessage(f"⚠️ Skipping malformed per-option upload item: {e}")
                            continue
                written = save_rows_to_csv_stream(gen_per_option(), TRAINING_DIR / f"_upload_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", chunk_size=5000)
                if written:
                    df_chunk = pd.read_csv(TRAINING_DIR / f"_upload_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    append_accumulated_data_append_only(df_chunk)
                    # cleanup handled by save_rows_to_csv_stream
            else:
                # Snapshot-level rows: we need to group them per osiKey -> build sequences
                logger.logMessage("Upload detected snapshot-level rows. Grouping by osiKey using temporary SQLite table (streaming)...")
                # create temp sqlite db to bulk insert snapshots (avoids keeping all in memory)
                tmp_sqlite = TRAINING_DIR / f"_upload_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                conn = sqlite3.connect(tmp_sqlite)
                cur = conn.cursor()
                # create table matching snapshot schema minimally: osiKey, timestamp, and feature columns as TEXT (we'll store values)
                cols = ["osiKey TEXT", "timestamp TEXT"] + [f"{c} TEXT" for c in FEATURE_COLUMNS]  # store as TEXT, cast later
                cur.execute(f"CREATE TABLE snaps ({', '.join(cols)})")
                conn.commit()
                # insert rows streamed
                insert_sql = f"INSERT INTO snaps VALUES ({', '.join(['?'] * (2 + len(FEATURE_COLUMNS)))})"
                inserted = 0
                with open(tmp_upload, "rb") as f_in:
                    for item in ijson.items(f_in, "item"):
                        try:
                            osi = item.get("osiKey")
                            ts = item.get("timestamp")
                            vals = [str(item.get(c, "")) for c in FEATURE_COLUMNS]
                            cur.execute(insert_sql, [osi, ts] + vals)
                            inserted += 1
                            if inserted % LOG_EVERY_N == 0:
                                conn.commit()
                                logger.logMessage(f"Inserted {inserted} snapshot rows into temp DB")
                        except Exception as e:
                            logger.logMessage(f"⚠️ Skipping malformed snapshot upload item: {e}")
                            continue
                conn.commit()
                # Now read grouped sequences from sqlite in streaming fashion using SQL (order by osiKey, timestamp)
                q = "SELECT * FROM snaps ORDER BY osiKey, timestamp"
                # use pandas read_sql_query with chunksize to avoid memory spike
                seq_rows = []
                last_osi = None
                processed = 0
                chunk_iter = pd.read_sql_query(q, conn, chunksize=50_000)
                def grouped_generator():
                    nonlocal processed
                    buffer_by_osi = {}
                    for chunk_df in chunk_iter:
                        # each chunk_df has columns: osiKey, timestamp, FEATURE_COLUMNS as text
                        for _, r in chunk_df.iterrows():
                            osi = r["osiKey"]
                            if osi is None:
                                continue
                            # build snapshot
                            snap = {"timestamp": r["timestamp"]}
                            for c in FEATURE_COLUMNS:
                                raw = r[c]
                                try:
                                    snap[c] = float(raw) if raw not in (None, "", "None") else 0.0
                                except Exception:
                                    snap[c] = 0.0
                            if osi not in buffer_by_osi:
                                buffer_by_osi[osi] = []
                            buffer_by_osi[osi].append(snap)
                            # flush if buffer too large (many OSIs)
                            if len(buffer_by_osi) > 50000:
                                # flush some OSIs
                                for k in list(buffer_by_osi.keys())[:10000]:
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
                                        logger.logMessage(f"Grouped and processed {processed} osiKeys from temp DB")
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
                            logger.logMessage(f"Grouped and processed {processed} osiKeys from temp DB")
                        yield row
                # stream to CSV
                written = save_rows_to_csv_stream(grouped_generator(), TRAINING_DIR / f"_upload_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", chunk_size=5000)
                if written:
                    df_chunk = pd.read_csv(TRAINING_DIR / f"_upload_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    append_accumulated_data_append_only(df_chunk)
                conn.close()
                tmp_sqlite.unlink(missing_ok=True)

        tmp_upload.unlink(missing_ok=True)
        logger.logMessage("✅ Upload processing complete")

        # Optional: auto_train
        if auto_train:
            res = train_hybrid_model_streamed(ACCUMULATED_DATA_PATH, batch_size=BATCH_SIZE, throttle_delay=0.0, device=DEVICE)
            return res

        return jsonable_encoder(to_native_types({"status": "appended"}))

    except Exception as e:
        logger.logMessage(f"Upload error: {e}")
        return jsonable_encoder(to_native_types({"status": "error", "message": str(e)}))

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
class BacktestParams(BaseModel):
    entry_threshold: float = 0.05
    capital_per_trade: float = 1000.0

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

@app.post("/backtest")
def backtest_endpoint(params: BacktestParams):
    try:
        if not ACCUMULATED_DATA_PATH.exists():
            return jsonable_encoder(to_native_types({"status": "error", "message": "No accumulated data found."}))
        res = backtest_on_csv(ACCUMULATED_DATA_PATH, entry_threshold=params.entry_threshold, capital_per_trade=params.capital_per_trade)
        return jsonable_encoder(to_native_types(res))
    except Exception as e:
        logger.logMessage(f"Backtest error: {e}")
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


logger = getLogger()
router = APIRouter()

@router.get("/evaluate")
def evaluate_endpoint(batch_size: int = 128):
    try:
        df = load_accumulated_training_csvs(chunk_size=batch_size)
        metrics = evaluate_model(df, batch_size=batch_size)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.logMessage(f"Evaluate error: {e}")
        return {"status": "error", "message": str(e)}

router = APIRouter()

ACCUMULATED_DATA_PATH = Path("training/accumulated_training.csv")

@router.post("/backtest/run")
def api_backtest(batch_size: int = 256):
    """
    Launch a streaming, low-memory backtest.
    Returns immediately with basic info; progress stored in backtest_status.json.
    """
    if not ACCUMULATED_DATA_PATH.exists():
        return {"status": "error", "message": "No accumulated training dataset found."}

    # Kick off the backtest synchronously for now
    results = run_backtest_streaming(ACCUMULATED_DATA_PATH, batch_size=batch_size)
    return {"status": "complete", "results": results}


@router.get("/backtest/status")
def api_backtest_status():
    if BACKTEST_STATUS_PATH.exists():
        return json.loads(BACKTEST_STATUS_PATH.read_text())
    return {"status": "none"}
