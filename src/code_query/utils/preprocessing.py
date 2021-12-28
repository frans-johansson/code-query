"""
Preprocesses the data downloaded by the download_data.py script for
more convenient usage during training and evaluation. Additional
processing may be required on a per-model basis. This module only
handles common preprocessing for all models.

The training, testing and validation splits from the raw data will
be accumulated into a single `corpus` file, which means new splits
will have to be generated when used for training.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from itertools import chain
from functools import partial

import fasttext
import pycountry

from code_query.config import DATA
from code_query.utils.helpers import get_lang_dir, get_model_dir
from code_query.utils.serialize import jsonl_gzip_load, jsonl_gzip_dump
from code_query.utils.logging import get_logger


logger = get_logger("preprocessing")


def docstring_is_language(
        sample: Dict[str, Any],
        model: fasttext.FastText,
        language_labels: Set[str],
        threshold: float=0.5
    ) -> bool:
    """
    Predicate function determining if the docstring of a given sample
    is likely to be one of the given languages using a fastText model.
    If any of the given languages are identified with confidence greater
    than the given threshold, the predicate returns `True`. 
    
    Note that the language labels need to be formated as `__label__{lang}`
    where `{lang}` is the two-letter code of that language. E.g. English
    would be encoded as `__label__en`.
    """
    docstring = sample["docstring"].replace("\n", "")  # Fasttext dislikes newlines
    preds, _ = model.predict(docstring, threshold=threshold)
    matches = set(preds).intersection(language_labels)
    return len(matches) > 0
    

def process_evaluation_data(code_lang: str) -> None:
    """
    Reads and exctracts relavant data from the {language}_dedupe_definitions_v2.pkl
    file for model evaluation. The data is saved as eval.jsonl.gz in the configured
    output directory for the given programming language.
    """
    out_dir = get_lang_dir(code_lang)
    out_dir.mkdir(exist_ok=True, parents=True)
    raw_dir = Path(DATA.DIR.RAW)

    # Read in the pickled data
    pkl_file = raw_dir / DATA.FILES.RAW_FULL_FILE.format(language=code_lang)
    with open(pkl_file, "rb") as source:
        logger.info("Loading %s" % pkl_file)
        eval_raw = pickle.load(source)
    
    # Extract and process relevant fields
    logger.info("Processing %s evaluation data" % code_lang)
    eval_data = [{
        "url": raw["url"],
        "identifier": raw["identifier"],
        "code": " ".join(raw["function_tokens"]),
        "query": " ".join(raw["docstring_tokens"]),
    } for raw in eval_raw]
    
    # Serialize
    jsonl_gzip_dump(eval_data, out_dir / "eval.jsonl.gz")


def process_training_data(
        code_lang: str,
        query_langs: Optional[List[str]],
    ) -> None:
    """
    Concatenates the raw training data splits into one corpus.jsonl.gz file
    for training, validation and testing. The data is saved in a corpus.jsonl.gz
    file in the configured output directory for the given programming language
    and optional natural language filter.
    """
    filter_predicate = lambda _: True
    if query_langs:
        # Set up natural language filter
        logger.info("Setting up fastText language model")
        model = fasttext.load_model(DATA.QUERY_LANGUAGE_FILTER.FASTTEXT_FILE)
        labels = {
            f"__label__{pycountry.languages.get(name=lang.capitalize()).alpha_2}" 
            for lang in query_langs
        }
        filter_predicate = partial(
            docstring_is_language,
            model=model,
            language_labels=labels
        )
    
    # Accumulate data to a single list
    raw_dir = Path(DATA.DIR.RAW)
    acc_data = []
    logger.info("Processing %s training, validation and testing data" % code_lang)
    for split in DATA.SPLIT_NAMES:
        split_dir = raw_dir / DATA.FILES.RAW_SPLITS.format(language=code_lang, split=split)
        # Chain together iterables over all files and extract relevant data
        split_chain = chain.from_iterable(jsonl_gzip_load(file) for file in split_dir.iterdir())
        # Append to acc_data
        acc_data += [{
            "url": raw["url"],
            "identifier": raw["func_name"],
            "code": " ".join(raw["code_tokens"]),
            "query": " ".join(raw["docstring_tokens"])
        } for raw in filter(filter_predicate, split_chain)]
    
    # Serialize the accumulated data into one file
    out_dir = get_model_dir(code_lang, query_langs)
    out_dir.mkdir(exist_ok=True, parents=True)
    jsonl_gzip_dump(acc_data, out_dir / "corpus.jsonl.gz")
