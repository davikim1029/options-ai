# main.py
import sys
import os
import subprocess
import signal
import time
from pathlib import Path
import psutil
import shutil
from logger.logger_singleton import getLogger
from logging import FileHandler
from constants import UVICORN_PORT, TRAINING_DIR
from utils.testing import copy_all_snapshot_data

# -----------------------------
# MLP imports (refactored)
# -----------------------------
from mlp.mlp_pipeline import run_pipeline
from mlp.mlp_backtester_core import run_backtest_permutations
from mlp.mlp_upload_api import upload_training_data

# -----------------------------
# Server & PID setup
# -----------------------------
PID_FILE = Path("server.ai_model_server.pid")
MODEL_SERVER_SCRIPT = "ai_model_service:app"  # FastAPI server module

UVICORN_CMD = [
    sys.executable, "-m", "uvicorn",
    MODEL_SERVER_SCRIPT,
    "--host", "0.0.0.0",
    f"--port={UVICORN_PORT}",
    "--no-access-log"
]

logger = getLogger()
log_file = Path("ai_model_server.log")
for handler in logger.logger.handlers:
    if isinstance(handler, FileHandler):
        log_file = Path(handler.baseFilename)
        break

# -----------------------------
# Server control functions
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

    total_cores = psutil.cpu_count(logical=True)
    use_cores = max(1, total_cores // 4)
    core_mask = ",".join(str(i) for i in range(use_cores))
    mem_limit_mb = 2048
    taskset_path = shutil.which("taskset")
    prefix_cmd = [taskset_path, "-c", core_mask] if taskset_path else []
    limit_cmd = ["bash", "-c", f"ulimit -v {mem_limit_mb * 1024}; exec " + " ".join(UVICORN_CMD)]
    full_cmd = prefix_cmd + limit_cmd

    with open(log_file, "a") as f:
        process = subprocess.Popen(
            full_cmd,
            stdout=f,
            stderr=f,
            preexec_fn=os.setpgrp
        )

    PID_FILE.write_text(str(process.pid))
    logger.logMessage(f"AI server started with PID {process.pid}, CPU cores: {core_mask}, memory limit: {mem_limit_mb} MB")

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

# -----------------------------
# CLI Menu
# -----------------------------
def main():
    while True:
        print("\nAI Model Manager")
        print("1) Start AI Server")
        print("2) Stop AI Server")
        print("3) Check Server Status")
        print("4) Upload MLP Data (auto-train)")
        print("5) Run MLP Pipeline (fetch & upload)")
        print("6) Run MLP Backtest")
        print("7) Populate Lifetime data with all snapshots")
        print("8) Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            start_server()
        elif choice == "2":
            stop_server()
        elif choice == "3":
            check_server()
        elif choice == "4":
            print("Uploading all MLP data and triggering training...")
            try:
                result = run_pipeline()  # Handles fetching + upload + auto_train
                print(result)
            except Exception as e:
                print(f"Error running MLP upload: {e}")
        elif choice == "5":
            print("Running MLP pipeline (fetch new unprocessed permutations + upload)...")
            try:
                result = run_pipeline()
                print(result)
            except Exception as e:
                print(f"Error running MLP pipeline: {e}")
        elif choice == "6":
            print("Running MLP backtest on latest permutations DB...")
            try:
                db_path = TRAINING_DIR / "option_permutations.db"
                result = run_backtest_permutations(db_path=db_path)
                print(result)
            except Exception as e:
                print(f"Error during MLP backtest: {e}")
        elif choice == "7":
            print("Populating lifetime data with all snapshots...")
            try:
                copy_all_snapshot_data()
            except Exception as e:
                print(f"Error populating snapshots: {e}")
        elif choice == "8":
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
