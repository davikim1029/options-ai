from pathlib import Path
import torch
from enum import Enum
import os
# -----------------------------
# Paths & Constants
# -----------------------------
UVICORN_PORT = 8100

MAX_SERVER_MEMORY_MB = 2048
CPU_CORE_FRACTION = 0.25  # use 25% of total cores


MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3
RESTART_DELAY = 2  # seconds before auto-restart

MIN_NEW_OPTIONS = 20  # number of new completed options before creating a file

class TrainerType(Enum):
    MLP = 0
    SGD = 1
    
    
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"
MODEL_DIR = BASE_DIR / "models" / "versions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(here, "..", "option-file-server", "database", "options.db"))

def get_model_path(type:TrainerType):
    return MODEL_DIR / f"{type.name}_current_model.pkl"


def get_scalar_path(type:TrainerType):
    return MODEL_DIR / f"{type.name}_current_scaler.pkl"


ACCUMULATED_DATA_PATH = TRAINING_DIR / "accumulated_training.csv"  # stores flattened first-snapshot + sequence column (json)

FEATURE_COLUMNS = [
    "optionType","strikePrice","lastPrice","bid","ask","bidSize","askSize",
    "volume","openInterest","nearPrice","inTheMoney","delta","gamma","theta",
    "vega","rho","iv","spread","midPrice","moneyness","daysToExpiration"
]
TARGET_COLUMNS = ["predicted_return","predicted_hold_time"]

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

