"""
Various helper functions
"""
from typing import Optional, List
from pathlib import Path
import requests

from tqdm import tqdm
import fasttext

from code_query.config import DATA


def get_identification_model() -> fasttext.FastText:
    """
    Returns a fasttext language identification model from
    the configured model file location. Will download the
    model if not already present.
    """
    model_file = Path(DATA.QUERY_LANGUAGE_FILTER.FASTTEXT_FILE)
    if model_file.exists():
        return fasttext.load_model(model_file)
    # Download is needed
    download_file(
        DATA.QUERY_LANGUAGE_FILTER.FASTTEXT_URL,
        model_file,
        description="Downloading FastText model"
    )
    return fasttext.load_model(model_file)


def download_file(url: str, output_path: Path, description="Downloading") -> None:
    """
    Wraps HTTP file downloads in a tqdm progress bar.
    """
    with requests.get(url, stream=True) as req, open(output_path, "wb") as file:
        size = int(req.headers["Content-length"])
        with tqdm(
                desc=description,
                total=size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as progress_bar:
            for chunk in req.iter_content(chunk_size=1024):
                file.write(chunk)
                progress_bar.update(1024)


def get_lang_dir(code_lang: str) -> Path:
    """
    Returns the default configured data directory for a given programming language
    """
    return Path(DATA.DIR.FINAL.format(language=code_lang))


def get_model_dir(code_lang: str, query_langs: Optional[List[str]]) -> Path:
    """
    Returns the default configured data directory for a given programming language
    filtered on an optional set of natural languages.
    """
    lang_dir = get_lang_dir(code_lang)
    nl_dir = "_".join(query_langs) if query_langs else DATA.QUERY_LANGUAGE_FILTER.DEFAULT_DIR
    return lang_dir / nl_dir
