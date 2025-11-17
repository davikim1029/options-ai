# mlp_upload_api.py
import uuid
import time
import sqlite3
from datetime import datetime
from fastapi import UploadFile, File, Form, WebSocket, APIRouter
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import pandas as pd
import numpy as np
import asyncio
from logger.logger_singleton import getLogger
from constants import TRAINING_DIR, LOG_EVERY_N, FEATURE_COLUMNS, TARGET_COLUMNS, DEVICE, BATCH_SIZE
from mlp_trainer import train_mlp_from_permutations

logger = getLogger()
router = APIRouter()

# -------------------------
# In-memory store for progress
# -------------------------
upload_progress = {}  # key: upload_id, value: dict with processed counts etc.

# -------------------------
# WebSocket progress endpoint
# -------------------------
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


# -------------------------
# Helpers
# -------------------------
def ensure_training_dir():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def write_rows_to_db(db_path: Path, rows_iterable, chunk_size: int = 5000):
    """Stream rows into option_permutations table safely."""
    ensure_training_dir()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Build insert placeholders
    sample_row = next(iter(rows_iterable), None)
    if not sample_row:
        return 0
    cols = list(sample_row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT OR REPLACE INTO option_permutations ({', '.join(cols)}) VALUES ({placeholders})"

    buffer = []
    written = 0
    for row in [sample_row] + list(rows_iterable):
        buffer.append(tuple(row[c] for c in cols))
        if len(buffer) >= chunk_size:
            cur.executemany(insert_sql, buffer)
            conn.commit()
            written += len(buffer)
            buffer.clear()
    if buffer:
        cur.executemany(insert_sql, buffer)
        conn.commit()
        written += len(buffer)
    conn.close()
    return written


# -------------------------
# Upload endpoint (MLP-ready)
# -------------------------
@router.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...), auto_train: bool = Form(default=True)):
    """
    Upload option permutation data for MLP training.
    Expects JSON array of dicts with FEATURE_COLUMNS + TARGET_COLUMNS.
    """
    upload_id = str(uuid.uuid4())
    upload_progress[upload_id] = {"processed": 0, "rate_per_s": 0, "message": "Starting", "start_time": time.time()}

    tmp_upload = TRAINING_DIR / f"_upload_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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
        # Read JSON in streaming fashion and yield rows
        # -------------------
        import ijson
        def gen_rows():
            processed = 0
            for item in ijson.items(open(tmp_upload, "rb"), "item"):
                try:
                    # Ensure all FEATURE_COLUMNS + TARGET_COLUMNS exist
                    row = {c: float(item.get(c, 0.0)) for c in FEATURE_COLUMNS + TARGET_COLUMNS}
                    row["osiKey"] = item.get("osiKey", "")
                    row["optionType"] = int(item.get("optionType", 0))
                    processed += 1
                    if processed % LOG_EVERY_N == 0:
                        log_progress_ws(upload_id, processed, "Rows prepared for DB")
                    yield row
                except Exception as e:
                    logger.logMessage(f"⚠️ Skipping malformed upload item: {e}")
                    continue

        # -------------------
        # Write directly to DB
        # -------------------
        db_path = TRAINING_DIR / "option_permutations.db"
        written = write_rows_to_db(db_path, gen_rows())
        logger.logMessage(f"✅ Uploaded {written} rows into option_permutations table")

        # -------------------
        # Optional auto_train
        # -------------------
        if auto_train:
            res = train_mlp_from_permutations(db_path=db_path, batch_size=BATCH_SIZE, device=DEVICE)
            return res

        return jsonable_encoder({"status": "appended", "upload_id": upload_id, "rows_written": written})

    except Exception as e:
        logger.logMessage(f"❌ Upload error [Server Side]: {e}")
        return jsonable_encoder({"status": "error", "message": str(e)})
