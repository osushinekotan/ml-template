import logging
from pathlib import Path


def get_file_handler(log_filepath: str | Path) -> logging.FileHandler:
    """Create file handler and set it.

    Args:
    ----
        log_filepath (str | Path): log file path.

    Returns:
    -------
        logging.FileHandler: file handler.

    """
    file_handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    file_handler.setFormatter(formatter)

    return file_handler


def get_consol_handler() -> logging.StreamHandler:
    """Create console handler and set it.

    Returns
    -------
        logging.StreamHandler: handler for console output.

    """
    consol_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    consol_handler.setFormatter(formatter)

    return consol_handler
