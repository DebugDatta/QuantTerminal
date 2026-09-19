"""Utility decorators: @timer, @handle_errors."""

import time
import functools
import logging

logger = logging.getLogger(__name__)


def timer(func):
    """Log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def handle_errors(default=None, log=True):
    """Decorator that catches exceptions and returns a default value."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log:
                    logger.error(f"{func.__name__} failed: {e}")
                return default
        return wrapper
    return decorator
