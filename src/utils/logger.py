# No AI tools were used to generate any code in this script. 

import logging
from pathlib import Path

def get_logger(name, log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
