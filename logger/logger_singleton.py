from logger.logger import Logger

_logger = None
_registered = False

def getLogger():
    global _logger, _registered
    if _logger is None:
        _logger = Logger()
    return _logger
