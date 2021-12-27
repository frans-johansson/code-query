"""
Handles serializing to and from various formats
"""

from typing import Any, Dict, Iterable
from pathlib import Path
import gzip
import json
import yaml

from tqdm import tqdm


def jsonl_gzip_load(path: Path) -> Iterable:
    """
    Loads a .jsonl.gz file and returns an Iterable over the lines in the file

    Code adapted from: https://github.com/novoselrok/codesnippetsearch/blob/master/code_search/serialize.py
    under the MIT license
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path}", unit="rows"):
            yield json.loads(line)


def jsonl_gzip_dump(iterable: Iterable, path: Path, compress_level: int=5) -> None:
    """
    Serializes the items of an `Iterable` to lines in a .jsonl.gz file

    Code adapted from: https://github.com/novoselrok/codesnippetsearch/blob/master/code_search/serialize.py
    under the MIT license
    """
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=compress_level) as f:
        for item in tqdm(iterable, desc=f"Saving to {path}", unit="rows"):
            f.write(json.dumps(item) + "\n")


def yml_load(path: Path) -> Dict[str, Any]:
    """
    Loads a .yml file into a Python dictionary
    """
    with open(path, "rt") as source:
        content = yaml.safe_load(source)
    return content
