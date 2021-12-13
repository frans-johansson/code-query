"""
Preprocesses the data downloaded by the download_data.py script for
more convenient usage during training and evaluation. Additional
processing may be required on a per-model basis. This script only
handles common preprocessing for all models.
"""

import argparse
import pickle
import json
import gzip
from pathlib import Path
from typing import Iterable
from itertools import chain


def jsonl_gzip_load(path) -> Iterable:
    """
    Loads a .jsonl.gz file and returns an Iterable over the lines in the file
    Code adapted from: https://github.com/novoselrok/codesnippetsearch/blob/master/code_search/serialize.py
    under the MIT license
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def jsonl_gzip_serialize(iterable: Iterable, path: str, compress_level: int=5) -> None:
    """
    Serializes the items of an `Iterable` to lines in a .jsonl.gz file
    Code adapted from: https://github.com/novoselrok/codesnippetsearch/blob/master/code_search/serialize.py
    under the MIT license
    """
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=compress_level) as f:
        for item in iterable:
            f.write(json.dumps(item) + '\n')


def process_evaluation_data(language: str, raw_dir: Path, out_dir: Path) -> None:
    """
    Reads and exctracts relavant data from the {language}_dedupe_definitions_v2.pkl
    file and saves this data to `out_dir` as eval.jsonl.gz
    """
    # Read in the pickled data
    pkl_file = raw_dir / f"{language}_dedupe_definitions_v2.pkl"
    with open(pkl_file, "rb") as source:
        eval_raw = pickle.load(source)
    # Extract relevant fields
    eval_data = ({
        "url": raw["url"],
        "identifier": raw["identifier"],
        "code": raw["function"],
        "code_tokens": raw["function_tokens"],
        "query": raw["docstring_summary"],
        "query_tokens": raw["docstring_tokens"],
    } for raw in eval_raw)
    # Serialize
    jsonl_gzip_serialize(eval_data, out_dir / "eval.jsonl.gz")


def process_training_data(language: str, raw_dir: Path, out_dir: Path) -> None:
    """
    Concatenates the training data splits into individual train, test and valid
    .jsonl.gz files in `out_dir`
    """
    data_dir = raw_dir / language / "final/jsonl"
    for split in ("test", "train", "valid"):
        split_dir = data_dir / split
        # Chain together iterables over all files and extract relevant data
        split_chain = chain.from_iterable(jsonl_gzip_load(file) for file in split_dir.iterdir())
        split_data = ({
            "url": raw["url"],
            "identifier": raw["func_name"],
            "code": raw["code"],
            "code_tokens": raw["code_tokens"],
            "query": raw["docstring"],
            "query_tokens": raw["docstring_tokens"],
        } for raw in split_chain)
        # Serialize the chained results into one file
        jsonl_gzip_serialize(split_data, out_dir / f"{split}.jsonl.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess data")
    parser.add_argument("-r", "--raw-dir", type=Path, help="Where to read the raw data from")
    parser.add_argument("-d", "--out-dir", type=Path, help="Where put the processed data")
    parser.add_argument("-l", "--languages", choices=('python', 'javascript', 'java', 'ruby', 'php', 'go'), help="What languages to process data for", nargs="+")

    args = parser.parse_args()

    # Set up output directory
    args.out_dir.mkdir(exist_ok=True)

    # Parse data for all languages
    for language in args.languages:
        print("Processing data for", language)
        lang_dir = args.out_dir / language
        lang_dir.mkdir(exist_ok=True)
        process_evaluation_data(language, args.raw_dir, lang_dir)
        process_training_data(language, args.raw_dir, lang_dir)
