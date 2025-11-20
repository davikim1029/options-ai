# mlp_trainer.py
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import joblib
from pathlib import Path
from shared_options.log.logger_singleton import getLogger
from constants import (
    BATCH_SIZE, FEATURE_COLUMNS, TARGET_COLUMNS, DB_PATH,
    get_model_path, get_scalar_path, TrainerType
)

logger = getLogger()


# -----------------------------
# Stream option_permutations in chunks
# -----------------------------
def stream_permutation_rows(db_path: Path, batch_size: int = BATCH_SIZE):
    """Generator yielding rows in chunks from the option_permutations table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    last_rowid = 0
    while True:
        rows = conn.execute(
            "SELECT * FROM option_permutations WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
            (last_rowid, batch_size)
        ).fetchall()
        if not rows:
            break

        last_rowid = rows[-1]["rowid"]
        yield rows

    conn.close()


# -----------------------------
# Convert rows to numpy arrays
# -----------------------------
def rows_to_xy(rows, feature_cols=FEATURE_COLUMNS, target_cols=TARGET_COLUMNS):
    X, y = [], []
    for row in rows:
        row_dict = dict(row)
        # Features
        X.append([float(row_dict.get(f, 0.0)) for f in feature_cols])
        # Targets
        y.append([float(row_dict.get(t, 0.0)) for t in target_cols])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# -----------------------------
# Incremental MLP training
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

        # Incremental fit (assuming first target is hold_time or main prediction)
        mlp_model.partial_fit(X_scaled, y_chunk[:, 0])

        total_rows += len(X_chunk)
        logger.logMessage(f"Trained on chunk of {len(X_chunk)} rows | Total rows processed: {total_rows}")

    # Save model & scaler
    model_path = get_model_path(TrainerType.MLP)
    scaler_path = get_scalar_path(TrainerType.MLP)
    joblib.dump(mlp_model, model_path)
    joblib.dump(scaler, scaler_path)

    logger.logMessage(f"✅ Training complete. Model saved to {model_path}, scaler saved to {scaler_path}")
    return mlp_model, scaler
