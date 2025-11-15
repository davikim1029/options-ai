# backtester_api.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import json
from backtester_core import run_backtest_streaming, BACKTEST_STATUS_PATH, _backtest_lock

router = APIRouter(prefix="/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    csv_path: str = "training/accumulated_training.csv"
    batch_size: int = 128

@router.post("/start")
def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    csv_file = Path(request.csv_path)
    if not csv_file.exists():
        return {"status": "error", "message": f"CSV file not found: {csv_file}"}

    if not _backtest_lock.acquire(blocking=False):
        return {"status": "error", "message": "A backtest is already running."}

    def _run():
        try:
            run_backtest_streaming(csv_file, batch_size=request.batch_size)
        finally:
            _backtest_lock.release()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Backtest running in background."}

@router.get("/status")
def get_backtest_status():
    if not BACKTEST_STATUS_PATH.exists():
        return {"status": "idle", "progress": 0, "results": None}
    
    try:
        with BACKTEST_STATUS_PATH.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {"status": "error", "progress": 0, "results": None}
    
    return data
