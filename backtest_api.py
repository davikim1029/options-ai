# ai_model_service/backtest_api.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import json
from threading import Lock
from backtester import run_backtest_streaming, BACKTEST_STATUS_PATH

router = APIRouter(prefix="/backtest", tags=["Backtest"])

# Thread-safety lock to prevent multiple backtests at once
_backtest_lock = Lock()

class BacktestRequest(BaseModel):
    csv_path: str = "training/accumulated_training.csv"
    batch_size: int = 128

@router.post("/start")
def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Start a new backtest in the background.
    """
    if not _backtest_lock.acquire(blocking=False):
        return {"status": "error", "message": "A backtest is already running."}

    csv_file = Path(request.csv_path)
    if not csv_file.exists():
        _backtest_lock.release()
        return {"status": "error", "message": f"CSV file not found: {csv_file}"}

    # Run backtest in background
    def _run():
        try:
            run_backtest_streaming(csv_file, batch_size=request.batch_size)
        finally:
            _backtest_lock.release()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Backtest running in background."}

@router.get("/status")
def get_backtest_status():
    """
    Get current backtest progress and results.
    """
    if not BACKTEST_STATUS_PATH.exists():
        return {"status": "idle", "progress": 0, "results": None}
    
    with BACKTEST_STATUS_PATH.open("r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"status": "error", "progress": 0, "results": None}
    
    return data
