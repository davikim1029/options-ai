import sys
import os
import subprocess
import signal
import time
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
import requests
from utils.mock_data import generate_sample_training_file  # your mock data generator

# -----------------------------
# Paths & Constants
# -----------------------------
PID_FILE = Path("ai_model_server.pid")
LOG_FILE = Path("ai_model_server.log")
TRAINING_DIR = Path("training")
MODEL_SERVER_SCRIPT = "ai_model_service:app"  # FastAPI server module
UVICORN_PORT = 8100

MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3
RESTART_DELAY = 2  # seconds before auto-restart

UVICORN_CMD = [
    sys.executable, "-m", "uvicorn",
    MODEL_SERVER_SCRIPT,
    "--host", "0.0.0.0",
    f"--port={UVICORN_PORT}",
    "--log-level", "info"
]

# -----------------------------
# Setup Logger
# -----------------------------
logger = logging.getLogger("AIModelManager")
logger.setLevel(logging.INFO)
LOG_FILE.parent.mkdir(exist_ok=True)
handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------
# Server Control Functions
# -----------------------------
def is_server_running():
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError):
        PID_FILE.unlink(missing_ok=True)
        return False

def start_server():
    if is_server_running():
        print("AI server already running.")
        return
    with LOG_FILE.open("a") as log_file:
        process = subprocess.Popen(UVICORN_CMD, stdout=log_file, stderr=log_file)
    PID_FILE.write_text(str(process.pid))
    print(f"AI server started with PID {process.pid}, logging to {LOG_FILE}")

def stop_server():
    if not PID_FILE.exists():
        print("AI server not running.")
        return
    pid = int(PID_FILE.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
        time.sleep(1)
    except ProcessLookupError:
        print(f"No process with PID {pid} found.")
    PID_FILE.unlink(missing_ok=True)
    print("AI server stopped.")

def check_server():
    if is_server_running():
        pid = int(PID_FILE.read_text())
        print(f"AI server running with PID {pid}")
    else:
        print("AI server is not running.")

def tail_log(n=10):
    if not LOG_FILE.exists():
        print("Log file does not exist.")
        return
    with LOG_FILE.open("r") as f:
        last_lines = deque(f, maxlen=n)
    for line in last_lines:
        print(line, end='')

# -----------------------------
# API Interaction Functions
# -----------------------------
def upload_sample_data(auto_train=True):
    TRAINING_DIR.mkdir(exist_ok=True)
    sample_file = generate_sample_training_file(TRAINING_DIR)
    url = f"http://127.0.0.1:{UVICORN_PORT}/train/upload"
    with open(sample_file, "rb") as f:
        files = {"file": f}
        data = {"auto_train": str(auto_train).lower()}
        try:
            resp = requests.post(url, files=files, data=data)
            print(resp.json())
        except requests.exceptions.RequestException as e:
            print(f"Error uploading sample data: {e}")

def upload_custom_csv(file_path: Path, auto_train=True):
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return
    url = f"http://127.0.0.1:{UVICORN_PORT}/train/upload"
    with file_path.open("rb") as f:
        files = {"file": f}
        data = {"auto_train": str(auto_train).lower()}
        try:
            resp = requests.post(url, files=files, data=data)
            print(resp.json())
        except requests.exceptions.RequestException as e:
            print(f"Error uploading CSV: {e}")

def train_accumulated_data():
    url = f"http://127.0.0.1:{UVICORN_PORT}/train"
    try:
        resp = requests.post(url)
        print(resp.json())
    except requests.exceptions.RequestException as e:
        print(f"Error training accumulated data: {e}")

def run_prediction():
    url = f"http://127.0.0.1:{UVICORN_PORT}/predict"
    payload = {
        "features": {
            "optionType": 1, "strikePrice": 50.0, "lastPrice": 52.0, "bid": 51.5,
            "ask": 52.5, "bidSize": 10, "askSize": 15, "volume": 100, "openInterest": 200,
            "nearPrice": 50.0, "inTheMoney": 1, "delta": 0.5, "gamma": 0.01, "theta": -0.02,
            "vega": 0.03, "rho": 0.01, "iv": 0.25, "spread": 1.0, "midPrice": 52.0,
            "moneyness": 0.96, "daysToExpiration": 10.0
        }
    }
    try:
        resp = requests.post(url, json=payload)
        print(resp.json())
    except requests.exceptions.RequestException as e:
        print(f"Error running prediction: {e}")

# -----------------------------
# CLI Menu
# -----------------------------
def main():
    while True:
        print("\nAI Model Manager")
        print("1) Start AI Server")
        print("2) Stop AI Server")
        print("3) Check Server Status")
        print("4) Tail last 10 log lines")
        print("5) Upload sample training data")
        print("6) Upload custom CSV")
        print("7) Train accumulated data")
        print("8) Run test prediction")
        print("9) Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            start_server()
        elif choice == "2":
            stop_server()
        elif choice == "3":
            check_server()
        elif choice == "4":
            tail_log()
        elif choice == "5":
            upload_sample_data()
        elif choice == "6":
            file_path = input("Enter path to CSV file: ").strip()
            upload_custom_csv(Path(file_path))
        elif choice == "7":
            train_accumulated_data()
        elif choice == "8":
            run_prediction()
        elif choice == "9":
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
