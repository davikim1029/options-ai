# shared_utils/io_utils.py
import pandas as pd
import time
from pathlib import Path
from shared_options.log.logger_singleton import getLogger
import ast
import numpy as np
import json
from constants import get_model_path,get_scalar_path


def save_csv_safely(data, output_path, chunksize=25_000, delay=0.2, logger=None):
    """
    Save a DataFrame or list of dictionaries to CSV in chunks to reduce I/O pressure.
    
    Parameters
    ----------
    data : pd.DataFrame or list of dict
        The data to save.
    output_path : str or Path
        Path to save the CSV file.
    chunksize : int
        Number of rows per chunk.
    delay : float
        Seconds to wait between writing chunks.
    logger : optional
        Logger object with .logMessage() method.
    """
    
    if logger is None:
        logger = getLogger()
    # Convert list of dicts to DataFrame if needed
    if isinstance(data, list):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"Unsupported data type: {type(data)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_rows = len(data)
    if logger:
        logger.logMessage(f"Saving {total_rows} rows to {output_path} in chunks of {chunksize}...")

    # Write CSV in chunks
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        for start in range(0, total_rows, chunksize):
            end = min(start + chunksize, total_rows)
            chunk = data.iloc[start:end]
            header = start == 0  # only write header for first chunk
            chunk.to_csv(f, index=False, header=header, mode='a')
            if logger:
                logger.logMessage(f"Wrote rows {start}–{end} / {total_rows}")
            time.sleep(delay)
    
    if logger:
        logger.logMessage(f"CSV save completed: {output_path}")
    return output_path


# -----------------------------
# Utility helpers
# -----------------------------
from decimal import Decimal

def to_native_types(obj):
    """Convert numpy scalars, arrays, Decimals, and tuples to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_native_types(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_native_types(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, Decimal)):
        return float(obj)

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    return obj



def safe_literal_eval(s):
    """Safely parse python literal from string (used for sequence stored as string)."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def write_sequence_streaming(path, generator, logger=None):
    with open(path, "w") as f:
        f.write("[")
        first = True
        cnt = 0

        for item in generator:
            cnt += 1
            if logger and cnt % 1000 == 0:
                logger.logMessage(f"JSON stream wrote {cnt} items")

            # convert numpy → native types here
            item = to_native_types(item)

            if not first:
                f.write(",")

            json.dump(item, f)
            first = False

        f.write("]")


# evaluation.py
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from constants import TrainerType

import joblib
from pathlib import Path
from datetime import datetime
import shutil

MODEL_STORE = Path("model_store")
CURRENT_DIR = MODEL_STORE / "current"
ARCHIVE_DIR = MODEL_STORE / "archive"

CURRENT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, scaler, model_type:TrainerType=TrainerType.MLP, keep_last_n=10):
    """
    Save model/scaler with timestamp, update current, archive old ones.
    
    Args:
        model: sklearn / MLP model object
        scaler: scaler object
        model_type: "mlp" or "sgd"
        keep_last_n: number of archived versions to retain
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = ARCHIVE_DIR / f"{model_type}_model_{timestamp}.pkl"
    scaler_file = ARCHIVE_DIR / f"{model_type}_scaler_{timestamp}.pkl"

    # Save to archive first
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    # Update "current" symlinks or replace files
    current_model = get_model_path(model_type)
    current_scaler = get_scalar_path(model_type)
    shutil.copy(model_file, current_model)
    shutil.copy(scaler_file, current_scaler)

    # Clean old archives
    archived_models = sorted(ARCHIVE_DIR.glob(f"{model_type}_model_*.pkl"), reverse=True)
    archived_scalers = sorted(ARCHIVE_DIR.glob(f"{model_type}_scaler_*.pkl"), reverse=True)

    for old_file in archived_models[keep_last_n:]:
        old_file.unlink()
    for old_file in archived_scalers[keep_last_n:]:
        old_file.unlink()

    print(f"[{timestamp}] Saved {model_type} model & scaler -> current, archived {len(archived_models)} versions")
    return current_model, current_scaler

def load_model(type:TrainerType=TrainerType.MLP):
    """
    Load the current model/scaler.
    """
    current_model = get_model_path(type)
    current_scaler = get_scalar_path(type)
    if not current_model.exists() or not current_scaler.exists():
        raise FileNotFoundError(f"No current {TrainerType.MLP.name} model/scaler found.")
    model = joblib.load(current_model)
    scaler = joblib.load(current_scaler)
    return model, scaler
