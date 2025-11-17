import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import joblib
from pathlib import Path
from shared_options.log.logger_singleton import getLogger
from constants import (BATCH_SIZE,FEATURE_COLUMNS,TARGET_COLUMNS,DB_PATH,get_model_path,get_scalar_path, TrainerType)

logger = getLogger()

# -----------------------------
# Stream permutations in chunks
# -----------------------------
def stream_permutation_rows(db_path, batch_size=BATCH_SIZE):
    """Generator yielding chunks of rows from option_permutations."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Use rowid to paginate
    last_rowid = 0
    while True:
        c.execute(f"""
            SELECT *
            FROM option_permutations
            WHERE rowid > ?
            ORDER BY rowid ASC
            LIMIT ?
        """, (last_rowid, batch_size))
        rows = c.fetchall()
        if not rows:
            break
        last_rowid = rows[-1][0]  # rowid of last row in chunk
        yield rows

    conn.close()

# -----------------------------
# Map rows to X, y arrays
# -----------------------------
def rows_to_xy(rows, feature_cols=FEATURE_COLUMNS, target_cols=TARGET_COLUMNS):
    """Convert SQLite rows to NumPy arrays for training."""
    # Fetch column names from first row
    col_names = [description[0] for description in rows[0]._fields] if hasattr(rows[0], "_fields") else None
    X = []
    y = []

    for r in rows:
        if col_names:
            X.append([r[col_names.index(f)] for f in feature_cols])
            y.append([r[col_names.index(t)] for t in target_cols])
        else:
            # fallback: assume table order matches FEATURE_COLS + TARGET_COLS
            X.append([r[FEATURE_COLUMNS.index(f)+1] for f in feature_cols])  # +1 to skip osiKey
            y.append([r[len(FEATURE_COLUMNS)+i+3] for i, t in enumerate(target_cols)])  # crude fallback
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# -----------------------------
# Training loop
# -----------------------------
def train_mlp_from_permutations(db_path=DB_PATH, batch_size=BATCH_SIZE):
    logger.logMessage("🚀 Starting incremental MLP training on option_permutations...")

    scaler = StandardScaler()
    mlp_model = SGDRegressor(max_iter=1000, tol=1e-3, warm_start=True)

    first_chunk = True
    total_rows = 0

    for chunk in stream_permutation_rows(db_path, batch_size=batch_size):
        X_chunk, y_chunk = rows_to_xy(chunk)

        # Scale features
        if first_chunk:
            X_scaled = scaler.fit_transform(X_chunk)
            first_chunk = False
        else:
            X_scaled = scaler.transform(X_chunk)

        # Incremental fit
        mlp_model.partial_fit(X_scaled, y_chunk[:, 0])  # assuming hold_time target

        total_rows += len(X_chunk)
        logger.logMessage(f"Trained on chunk of {len(X_chunk)} rows | Total rows processed: {total_rows}")

    # Save model & scaler
    model_path = get_model_path(TrainerType.MLP)
    scaler_path = get_scalar_path(TrainerType.MLP)
    joblib.dump(mlp_model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.logMessage(f"✅ Training complete. Model saved to {model_path}, scaler saved to {scaler_path}")
    return mlp_model, scaler