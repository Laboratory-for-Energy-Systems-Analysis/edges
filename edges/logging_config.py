from pathlib import Path
import logging
import logging.config


def setup_package_logging(
    log_path: Path = Path("edges.log"), level: int = logging.INFO
) -> None:
    """
    Route all logs from the 'edges' package (and submodules) into one file.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "edges_file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_path),
                    "mode": "w",
                    "encoding": "utf-8",
                    "formatter": "standard",
                    "level": level,
                }
            },
            "loggers": {
                "edges": {
                    "handlers": ["edges_file"],
                    "level": level,
                    "propagate": False,
                }
            },
            "root": {"level": "WARNING", "handlers": []},
        }
    )
