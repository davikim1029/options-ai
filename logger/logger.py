# services/logging/logger.py
import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", prefix="log"):
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{prefix}_{today}.log")

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Remove old handlers
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)

        # File handler
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
        self.logger.addHandler(ch)

        # Keep references
        self._file_handler = fh
        self._console_handler = ch

    def logMessage(self, message, console=True, file=True):
        # Temporarily enable/disable handlers
        self._file_handler.setLevel(logging.INFO if file else logging.CRITICAL+1)
        self._console_handler.setLevel(logging.INFO if console else logging.CRITICAL+1)
        self.logger.info(message)
        self.flush()
        
    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def _log_exit(self, reason=None):
        self.log(f"Script terminated ({reason})")
