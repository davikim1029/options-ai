from pathlib import Path
import torch
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
