from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
from pathlib import Path
from backtester_core import run_backtest_streaming, _backtest_lock
from shared_options.log.logger_singleton import getLogger
import threading
import os
import traceback

router = APIRouter(prefix="/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    csv_path: str = "training/accumulated_training.csv"
    batch_size: int = 128

@router.post("/start")
def start_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    fastapi_request: Request
):
    logger = getLogger()
    pid = os.getpid()
    tid = threading.get_ident()
    req_id = getattr(fastapi_request.state, "request_id", "no-request")

    logger.logMessage(f"[request_id={req_id}] PID={pid} Thread={tid} Backtest Start API hit")

    csv_file = Path(request.csv_path)
    if not csv_file.exists():
        logger.logMessage(f"[request_id={req_id}] PID={pid} Thread={tid} CSV not found: {csv_file}")
        return {"status": "error", "message": f"CSV file not found: {csv_file}"}

    # Do NOT acquire any lock here!
    # The lock is handled fully inside run_backtest_streaming()

    def _run_backtest():
        tid_inner = threading.get_ident()
        try:
            logger.logMessage(f"[request_id={req_id}] PID={pid} Thread={tid_inner} Running backtest streaming")
            run_backtest_streaming(csv_file, batch_size=request.batch_size, fastapi_request=fastapi_request)
        except Exception:
            logger.logMessage(
                f"[request_id={req_id}] PID={pid} Thread={tid_inner} Exception during backtest:\n{traceback.format_exc()}",
            )

    background_tasks.add_task(_run_backtest)
    return {"status": "started", "message": "Backtest running in background."}


@router.get("/status")
def get_backtest_status():
    from backtester_core import BACKTEST_STATUS_PATH
    import json

    if not BACKTEST_STATUS_PATH.exists():
        return {"status": "idle", "progress": 0, "results": None}

    try:
        with BACKTEST_STATUS_PATH.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {"status": "error", "progress": 0, "results": None}

    return data
