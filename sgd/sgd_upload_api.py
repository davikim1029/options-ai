# ai_server.py  (FULL production-ready rewrite - drop-in replacement)
import uuid
import time
import sqlite3
import ijson
import pandas as pd
from datetime import datetime
from fastapi import UploadFile, File, Form,WebSocket,APIRouter
from fastapi.encoders import jsonable_encoder
from logger.logger_singleton import getLogger
from utils.utils import safe_literal_eval
from fastapi.encoders import jsonable_encoder
import asyncio
from pathlib import Path
from constants import TRAINING_DIR, LOG_EVERY_N, FEATURE_COLUMNS, ACCUMULATED_DATA_PATH,BATCH_SIZE,DEVICE,TARGET_COLUMNS
from sgd.sgd_training import compute_end_of_life_targets,train_hybrid_model_streamed
from utils.utils import to_native_types
import json
import numpy as np

logger = getLogger()
router = APIRouter()
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
# CSV helpers (streaming safe)
# -----------------------------
def ensure_training_dir():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)


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
