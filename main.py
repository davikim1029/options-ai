#main.py
import sys
import os
import subprocess
import signal
import time
from pathlib import Path
from collections import deque
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
import shutil
from logger.logger_singleton import getLogger
from logging import FileHandler
from pipeline import run_pipeline
from utils.mock_data import copy_all_snapshot_data

# -----------------------------
# Paths & Constants
# -----------------------------
PID_FILE = Path("ai_model_server.pid")
TRAINING_DIR = Path("training")
MODEL_SERVER_SCRIPT = "ai_model_service:app"  # FastAPI server module
UVICORN_PORT = 8100

MAX_SERVER_MEMORY_MB = 2048
CPU_CORE_FRACTION = 0.25  # use 25% of total cores


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
log_file = Path("ai_model_server.log")
for handler in logger.logger.handlers:
    if isinstance(handler, FileHandler):
        log_file = Path(handler.baseFilename)
        break # Assuming you only care about the first FileHandler found
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

    # Determine CPU cores
    total_cores = psutil.cpu_count(logical=True)
    use_cores = max(1, total_cores // 4)  # Use up to 25% of cores
    core_mask = ",".join(str(i) for i in range(use_cores))
    
    # Set memory limit (2 GB)
    mem_limit_mb = 2048

    # Check for 'taskset' availability
    taskset_path = shutil.which("taskset")
    if taskset_path:
        prefix_cmd = [taskset_path, "-c", core_mask]
    else:
        prefix_cmd = []

    # Prepare ulimit wrapper for memory
    limit_cmd = [
        "bash", "-c",
        f"ulimit -v {mem_limit_mb * 1024}; exec " + " ".join(UVICORN_CMD)
    ]

    full_cmd = prefix_cmd + limit_cmd

    with open(log_file, "a") as f:
        process = subprocess.Popen(
            full_cmd,
            stdout=f,
            stderr=f,
            preexec_fn=os.setpgrp  # allows later clean kill
        )

    PID_FILE.write_text(str(process.pid))
    logger.logMessage(
        f"AI server started with PID {process.pid}, "
        f"CPU cores: {core_mask}, memory limit: {mem_limit_mb} MB, log: {log_file}"
    )

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
    if not log_file.exists():
        print("Log file does not exist.")
        return
    with log_file.open("r") as f:
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
        print("9) Run Pipeline")
        print("10) Populate Lifetime data with all snapshots")
        print("11) Exit")
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
            run_pipeline()
        elif choice == "10":
            copy_all_snapshot_data()
        elif choice == "11":
            break
        elif choice == "12":
            start_backtest_cli()

        elif choice == "13":
            url = f"http://127.0.0.1:{UVICORN_PORT}/backtest/run"
            print(requests.post(url).json())

        elif choice == "14":
            url = f"http://127.0.0.1:{UVICORN_PORT}/backtest/status"
            print(requests.get(url).json())

        else:
            print("Invalid choice, try again.")


# -----------------------------
# Backtest CLI Integration
# -----------------------------
def start_backtest_cli():
    from fastapi import status
    import requests
    import json

    url_start = f"http://127.0.0.1:{UVICORN_PORT}/backtest/start"
    url_status = f"http://127.0.0.1:{UVICORN_PORT}/backtest/status"

    csv_path = input("Enter path to CSV file (default: training/accumulated_training.csv): ").strip()
    if not csv_path:
        csv_path = str(TRAINING_DIR / "accumulated_training.csv")

    try:
        resp = requests.post(url_start, json={"csv_path": csv_path, "batch_size": 128})
        data = resp.json()
        if data.get("status") != "started":
            logger.logMessage(f"Could not start backtest: {data.get('message')}")
            return
        logger.logMessage("Backtest started in background. Polling status...")

        # Poll status every 2 seconds until complete
        import time
        while True:
            resp_status = requests.get(url_status)
            status_data = resp_status.json()
            progress = status_data.get("progress", 0)
            state = status_data.get("status", "idle")
            print(f"\r[{state}] Progress: {progress}%", end="", flush=True)

            if state == "complete":
                print("\n✅ Backtest complete!")
                results = status_data.get("results", {})
                print(json.dumps(results, indent=2))
                logger.logMessage(f"Backtest results: {results}")
                break
            elif state == "error":
                print(f"\n❌ Backtest error: {status_data.get('results')}")
                break

            time.sleep(2)

    except requests.exceptions.RequestException as e:
        logger.logMessage(f"Error communicating with AI server: {e}")



if __name__ == "__main__":
    main()
