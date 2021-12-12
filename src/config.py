from pathlib import Path
from typing import List
from dotenv import dotenv_values

# Read in configuration from .env file
__config = dotenv_values("../config.env")

# Root dir should be the base for all relative paths 
__root_dir = Path.cwd().parent

# Expose formatted configuration data
LANGUAGES: List[str] = list(__config["LANGUAGES"].split())
RAW_DATA_DIR: Path = __root_dir / Path(__config["RAW_DIR"])
PROCESSED_DATA_DIR: Path = __root_dir / Path(__config["PROCESSED_DIR"])
