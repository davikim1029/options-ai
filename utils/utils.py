# shared_utils/io_utils.py
import pandas as pd
import time
from pathlib import Path
from shared_options.log.logger_singleton import getLogger
import ast
import numpy as np
import json


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
def to_native_types(obj):
    """Convert numpy scalars and arrays to native Python types."""
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
    """Safely parse python literal from string (used for sequence stored as string)."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def write_sequence_streaming(path, data, logger = None):
    with open(path, "w") as f:
        f.write("[")
        first = True
        cnt=0
        total = len(data)
        for item in data:
            cnt +=1 
            if logger:
                if cnt % 10000 == 0:
                    logger.logMessage(f"Processed items {cnt}/{total}")
            if not first:
                f.write(",")
            json.dump(item, f)
            first = False
        f.write("]")
