import logging
import os
from datetime import datetime

# Setup logger settings
def setup_logger(name=None, log_level="INFO", log_dir="logs", console=True, file=True):
    """
    Create and configure a logger

    Args:
        name: Logger name (use __name__ in your modules)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console: Whether to log to console
        file: Whether to log to file
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    if console:
        # For Unicode in Windows console, set PYTHONIOENCODING=utf-8 in your environment before running.
        try:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            # On Windows, fallback: disable console logging if encoding fails
            pass

    # File handler
    if file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Create a default logger
default_logger = setup_logger("app")


# Convenience functions
def info(msg):
    default_logger.info(msg)


def debug(msg):
    default_logger.debug(msg)


def warning(msg):
    default_logger.warning(msg)


def error(msg):
    default_logger.error(msg)


def critical(msg):
    default_logger.critical(msg)


# Usage examples:
if __name__ == "__main__":
    # Using default logger
    info("Application started")
    debug("Debug message")
    warning("This is a warning")
    error("This is an error")

    # Creating module-specific loggers
    db_logger = setup_logger("database")
    db_logger.info("Database connected")

    api_logger = setup_logger("api")
    api_logger.info("API server started")
