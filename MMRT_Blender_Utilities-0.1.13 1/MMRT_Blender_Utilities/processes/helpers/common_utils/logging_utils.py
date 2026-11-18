import logging
import functools, time

def setup_logger(verbosity: str):
    level_map = {
        "silent": logging.CRITICAL + 1,  # no logs at all
        "normal": logging.INFO,
        "debug": logging.DEBUG,
        "verbose": logging.NOTSET,
    }
    logging.basicConfig(
        level=level_map.get(verbosity, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbosity != "silent":
        logging.info(f"Logging set to {verbosity.upper()}")

def trace(func):
    """Decorator for function entry/exit logging when verbosity=verbose."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if logging.getLogger().level <= logging.NOTSET:
            logging.log(logging.NOTSET, f"→ Enter {func.__qualname__}")
        t0 = time.time()
        result = func(*args, **kwargs)
        if logging.getLogger().level <= logging.NOTSET:
            logging.log(logging.NOTSET, f"← Exit {func.__qualname__} ({time.time() - t0:.3f}s)")
        return result
    return wrapper

def log_exceptions(func):
    """Decorator to log exceptions raised in the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__qualname__}: {e}", exc_info=True)
            raise
    return wrapper