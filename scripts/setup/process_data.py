"""
Preprocesses the data downloaded by the download_data.py script for
more convenient usage during training and evaluation. Additional
processing may be required on a per-model basis. This script only
handles common preprocessing for all models.

The training, testing and validation splits from the raw data will
be accumulated into a single `corpus` file, which means new splits
will have to be generated when used for training.
"""

import argparse
import pickle
import json
import gzip
from pathlib import Path
from typing import Callable, Dict, Any, Iterable, Optional, Set
from itertools import chain
from functools import partial

import fasttext
import pycountry
from tqdm import tqdm


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
        for item in tqdm(iterable, desc=f"Saving to {path}"):
            f.write(json.dumps(item) + '\n')


def docstring_is_language(
        sample: Dict[str, Any],
        model: fasttext.FastText,
        language_labels: Set[str]
    ) -> bool:
    """
    Predicate function determining if the docstring of a given sample
    is one of the given languages using a fasttext model. Note that
    the language labels need to be formated as `__label__{lang}` where
    `{lang}` is the two-letter code of that language.
    E.g. English would be encoded as `__label__en`.
    """
    docstring = sample["docstring"].replace("\n", "")  # Fasttext dislikes newlines
    preds, _ = model.predict(docstring)
    matches = set(preds).intersection(language_labels)
    return len(matches) > 0
    

def process_evaluation_data(language: str, raw_dir: Path, out_dir: Path) -> None:
    """
    Reads and exctracts relavant data from the {language}_dedupe_definitions_v2.pkl
    file and saves this data to `out_dir` as eval.jsonl.gz
    """
    # Read in the pickled data
    pkl_file = raw_dir / f"{language}_dedupe_definitions_v2.pkl"
    with open(pkl_file, "rb") as source:
        print(f"Loading {pkl_file}")
        eval_raw = pickle.load(source)
    # Extract relevant fields
    eval_data = []
    with tqdm() as progress_bar:
        progress_bar.set_description(f"{language} evaluation data")
        for raw in eval_raw:
            eval_data.append({
                "url": raw["url"],
                "identifier": raw["identifier"],
                "code": raw["function"],
                "code_tokens": raw["function_tokens"],
                "query": raw["docstring_summary"],
                "query_tokens": raw["docstring_tokens"],
            })
            progress_bar.update()
    # Serialize
    jsonl_gzip_serialize(eval_data, out_dir / "eval.jsonl.gz")


def process_training_data(
        language: str,
        raw_dir: Path,
        out_dir: Path,
        filter_predicate: Optional[Callable[[dict[str, Any]], bool]],
    ) -> None:
    """
    Concatenates the training data splits into one corpus.jsonl.gz file in `out_dir`.
    Optionally filter the data based on a given predicate.
    """
    if not filter_predicate:
        # Handle empty filter with a dummy predicate
        filter_predicate = lambda _: True
    
    data_dir = raw_dir / language / "final/jsonl"
    split_names = ("test", "train", "valid")
    # Accumulate data to a single list
    acc_data = []
    for split in split_names:
        split_dir = data_dir / split
        # Chain together iterables over all files and extract relevant data
        split_chain = chain.from_iterable(jsonl_gzip_load(file) for file in split_dir.iterdir())
        # Append to acc_data with a nice progress bar
        with tqdm() as progress_bar:
            progress_bar.set_description(str(split_dir))
            for raw in filter(filter_predicate, split_chain):
                acc_data.append({
                    "url": raw["url"],
                    "identifier": raw["func_name"],
                    "code": raw["code"],
                    "code_tokens": raw["code_tokens"],
                    "query": raw["docstring"],
                    "query_tokens": raw["docstring_tokens"],
                })
                progress_bar.update()
    # Serialize the accumulated data into one file
    jsonl_gzip_serialize(acc_data, out_dir / "corpus.jsonl.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess data")
    parser.add_argument("-r", "--raw-dir", type=Path, help="Where to read the raw data from")
    parser.add_argument("-d", "--out-dir", type=Path, help="Where put the processed data")
    parser.add_argument("-l", "--languages", nargs="+", choices=('python', 'javascript', 'java', 'ruby', 'php', 'go'), help="What programming languages to process data for")
    parser.add_argument("-n", "--natural-languages", nargs="+", required=False, help="What natural languages to keep in the output data (e.g. English, French, ...)")
    parser.add_argument("-f", "--fasttext-model-file", type=str, required=False, help="Where to find the downloaded fasttext model for natural language filtering")
    args = parser.parse_args()

    # Set up output directory
    args.out_dir.mkdir(exist_ok=True)

    # Set up natural lanaguage filter with fasttext
    filter_predicate = None
    if args.natural_languages:
        assert args.fasttext_model_file, "Please specify the location of the fasttext model to do natual language filtering"
        # Set up natural language filter
        model = fasttext.load_model(args.fasttext_model_file)
        labels = {
            f"__label__{pycountry.languages.get(name=lang.capitalize()).alpha_2}" 
            for lang in args.natural_languages
        }
        filter_predicate = partial(
            docstring_is_language,
            model=model,
            language_labels=labels)

    # Parse data for all languages
    for language in args.languages:
        lang_dir = args.out_dir / language
        lang_dir.mkdir(exist_ok=True)
        process_training_data(language, args.raw_dir, lang_dir, filter_predicate)
        process_evaluation_data(language, args.raw_dir, lang_dir)
