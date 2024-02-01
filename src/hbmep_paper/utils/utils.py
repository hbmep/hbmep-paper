import os
import logging

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(dir, fname, level=logging.INFO):
    fname = f"{fname.split('.')[0]}.log"
    dest = os.path.join(
        dir, fname
    )
    logging.basicConfig(
        format=FORMAT,
        level=level,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")
    return
