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
import pandas as pd
import numpy as np
from datetime import datetime
from logger.logger_singleton import getLogger

# -----------------------------
# Paths & Constants
# -----------------------------
PID_FILE = Path("ai_model_server.pid")
LOG_FILE = Path("logs/ai_model_server.log")
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
logger = getLogger()

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
        process = subprocess.Popen(
            UVICORN_CMD,
            stdout=log_file,
            stderr=log_file
        )
    PID_FILE.write_text(str(process.pid))
    logger.logMessage(f"AI server started with PID {process.pid}, logging to {LOG_FILE}")

def stop_server():
    if not PID_FILE.exists():
        print("AI server not running.")
        return
    pid = int(PID_FILE.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        logger.logMessage(f"Sent SIGTERM to PID {pid}")
        time.sleep(1)
    except ProcessLookupError:
        logger.logMessage(f"No process with PID {pid} found.")
    PID_FILE.unlink(missing_ok=True)
    logger.logMessage("AI server stopped.")

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
# Sample Data Generation
# -----------------------------
def generate_sample_training_file(training_dir: Path, n_rows=500):
    training_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = training_dir / f"sample_option_training_{timestamp}.csv"

    df = pd.DataFrame({
        "optionType": np.random.randint(0,2,size=n_rows),
        "strikePrice": np.random.uniform(10,100,size=n_rows),
        "lastPrice": np.random.uniform(10,100,size=n_rows),
        "bid": np.random.uniform(10,100,size=n_rows),
        "ask": np.random.uniform(10,100,size=n_rows),
        "bidSize": np.random.randint(1,100,size=n_rows),
        "askSize": np.random.randint(1,100,size=n_rows),
        "volume": np.random.randint(0,1000,size=n_rows),
        "openInterest": np.random.randint(0,1000,size=n_rows),
        "nearPrice": np.random.uniform(10,100,size=n_rows),
        "inTheMoney": np.random.randint(0,2,size=n_rows),
        "delta": np.random.uniform(-1,1,size=n_rows),
        "gamma": np.random.uniform(0,0.1,size=n_rows),
        "theta": np.random.uniform(-0.1,0,size=n_rows),
        "vega": np.random.uniform(0,0.1,size=n_rows),
        "rho": np.random.uniform(0,0.1,size=n_rows),
        "iv": np.random.uniform(0.1,1.0,size=n_rows),
        "spread": np.random.uniform(0,5,size=n_rows),
        "midPrice": np.random.uniform(10,100,size=n_rows),
        "moneyness": np.random.uniform(0,2,size=n_rows),
        "daysToExpiration": np.random.randint(1,30,size=n_rows),
        # target columns
        "predicted_return": np.random.uniform(-0.1,0.2,size=n_rows),
        "predicted_hold_days": np.random.randint(1,15,size=n_rows)
    })

    df.to_csv(file_path, index=False)
    logger.logMessage(f"✅ Generated sample training data: {file_path} ({n_rows} rows)")
    return file_path

# -----------------------------
# API Upload Functions
# -----------------------------
def upload_sample_data(auto_train=True):
    sample_file = generate_sample_training_file(TRAINING_DIR)
    url = f"http://127.0.0.1:{UVICORN_PORT}/train/upload"
    with open(sample_file, "rb") as f:
        files = {"file": f}
        data = {"auto_train": str(auto_train).lower()}
        try:
            resp = requests.post(url, files=files, data=data)
            logger.logMessage(resp.json())
        except requests.exceptions.RequestException as e:
            logger.logMessage(f"Error uploading sample data: {e}")

def upload_custom_csv():
    file_path = input("Enter path to CSV file: ").strip()
    if not Path(file_path).exists():
        logger.logMessage("File not found.")
        return
    url = f"http://127.0.0.1:{UVICORN_PORT}/train/upload"
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"auto_train": "true"}
        try:
            resp = requests.post(url, files=files, data=data)
            logger.logMessage(resp.json())
        except requests.exceptions.RequestException as e:
            logger.logMessage(f"Error uploading custom CSV: {e}")

def run_prediction_model():
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
        logger.logMessage(resp.json())
    except requests.exceptions.RequestException as e:
        logger.logMessage(f"Error making prediction: {e}")

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
            upload_custom_csv()
        elif choice == "7":
            url = f"http://127.0.0.1:{UVICORN_PORT}/train"
            try:
                resp = requests.post(url)
                print(resp.json())
            except requests.exceptions.RequestException as e:
                print(f"Error training accumulated data: {e}")
        elif choice == "8":
            run_prediction_model()
        elif choice == "9":
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
