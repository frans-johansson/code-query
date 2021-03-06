"""
Classes for downloading and processing CodeSearchNet (CSNet) data for training.
Most of the configuration options for these procedures can be controlled in the
data.yml, training.yml and models.yml configuration files.
"""
from collections import defaultdict
import zipfile
from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Iterator, List, Optional
from pathlib import Path
from functools import cached_property, partial
from itertools import islice, chain

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset, random_split
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

from code_query.config import DATA, TRAINING, MODELS
from code_query.utils.helpers import download_file, get_lang_dir, get_model_dir
from code_query.utils.logging import get_logger
from code_query.utils.serialize import jsonl_gzip_load
from code_query.utils.preprocessing import process_evaluation_data, process_training_data


TokenizerFunction = Callable[[str], torch.Tensor]

logger = get_logger("data")


def read_relevance_annotations(code_lang: str) -> Dict[str, Dict[str, float]]:
    """
    Reads the relevance annotations from a preconfigured .csv file and returns a
    dictionary with the following schema `{"query": {"url": [relevance_scores]}}`

    Code adapted from: https://github.com/github/CodeSearchNet/blob/master/src/relevanceeval.py
    under the MIT license.
    """
    relevance_annotations = pd.read_csv(DATA.RELEVANCE.ANNOTATIONS).query(f"Language == '{code_lang.capitalize()}'")
    per_query_language = relevance_annotations.pivot_table(
        index=['Query', 'GitHubUrl'], values='Relevance', aggfunc=np.mean)

    # Map language -> query -> url -> float
    relevances = defaultdict(lambda: defaultdict(float))  # type: Dict[str, Dict[str, Dict[str, float]]]
    for (query, url), relevance in per_query_language['Relevance'].items():
        relevances[query.lower()][url] = relevance
    return relevances


class CSNetDataManager(object):
    """
    Handles downloading, simple preprocessing and caching of the raw CSNet data. Provides
    iterators over the full training and evaluation corpus via class properties. Can be provided a `tiny`
    option to only yield iterators over a small subset of the data for debugging.
    """
    def __init__(
            self, 
            code_lang: str,
            query_langs: Optional[List[str]]=None,
            tiny: bool=False
        ) -> None:
        """
        Initiate CodeSearchNet data

        Args:
            code_lang (str): The programming language to download or load from disk.
            query_langs ([str], Optional): Specifies which natural languages to filter
                the queries on. Defaults to `None` which means no filtering is done.
            tiny (bool, Optional): Only load a small subset of data. Defaults to `False`. 
        """
        super().__init__()
        # Root directory of cached files
        self._model_dir = get_model_dir(code_lang, query_langs)
        self._root_dir = get_lang_dir(code_lang)
        self._query_langs = query_langs
        self._code_lang = code_lang
        self._tiny = tiny
    
    def _ensure_downloaded(self):
        """
        Takes care of downloading and processing CodeSearchNet data if needed.
        Does nothing if both downloaded and processed data is cached to file.
        """
        # Check if the raw data for the given language is available
        if not CSNetDataManager.has_downloaded(self._code_lang):
            logger.info("Did not find data for %s. Downloading now." % self._code_lang)
            CSNetDataManager.download_raw(self._code_lang)
        else:
            logger.info("Found downloaded data for %s" % self._code_lang)
        if not CSNetDataManager.has_processed(self._code_lang):
            logger.info("Did not find processed data for %s. Processing now." % self._code_lang)
            CSNetDataManager.process_raw(self._code_lang, self._query_langs)
        else:
            logger.info("Found processed data for %s" % self._code_lang)
    
    @cached_property
    def corpus(self) -> pd.DataFrame:
        """
        Provides the concatation of the processed raw training, testing and validation data 
        """
        logger.info("Reading corpus data")
        return self._load_as_dataframe(self._model_dir / "corpus.jsonl.gz")

    @cached_property
    def eval(self) -> pd.DataFrame:
        """
        Provides the complete processed raw dataset as given in the dedupe pickle file.
        """
        logger.info("Reading eval data")
        return self._load_as_dataframe(self._root_dir / "eval.jsonl.gz")

    def _load_as_dataframe(self, path: Path) -> pd.DataFrame:
        """
        Load a given .jsonl.gz file into a Pandas `DataFrame`.
        """
        self._ensure_downloaded()
        data_stream = jsonl_gzip_load(path)
        if self._tiny:
            data_stream = islice(data_stream, DATA.TINY_SIZE)
        return pd.DataFrame.from_records(data_stream)

    @staticmethod
    def has_downloaded(code_lang: str) -> bool:
        """
        Check if a given programming language has been downloaded.
        """
        dir = Path(DATA.DIR.RAW) / code_lang
        return dir.exists()

    @staticmethod
    def has_processed(code_lang: str, query_langs: Optional[List[str]]=None) -> bool:
        """
        Check if a given programming language has been preprocessed.
        """
        dir = get_model_dir(code_lang, query_langs)
        return dir.exists()

    @staticmethod
    def process_raw(code_lang: str, query_langs: Optional[List[str]]=None) -> None:
        """
        Handle preprocessing of a downloaded raw dataset for a given programming language
        and optional natural languages for query filtering.
        """
        assert CSNetDataManager.has_downloaded(code_lang), "Programming language not downloaded"
        process_training_data(
            code_lang,
            query_langs,
        )
        process_evaluation_data(
            code_lang
        )

    @staticmethod
    def download_raw(code_lang: str) -> None:
        """
        Download and unzip the raw data for a given programming language
        """
        assert not CSNetDataManager.has_downloaded(code_lang), "Programming language already downloaded"
        raw_path = Path(DATA.DIR.RAW)
        raw_path.mkdir(exist_ok=True, parents=True)
        url = DATA.URL.format(language=code_lang)
        zip_path = raw_path / f"{code_lang}.zip"
        # Download
        download_file(url, zip_path, description=f"Downloading {code_lang}.zip")
        # Unzip
        logger.info("Unzipping %s" % zip_path)
        with zipfile.ZipFile(zip_path, "r") as source:
            source.extractall(DATA.DIR.RAW)
        logger.info("Unzip done")
        # Clean up
        zip_path.unlink()


class CSNetDataset(Dataset):
    """
    A wrapper over a PyTorch TensorDataset over either the training or evaluation
    corpus supplied by the `CSNetDataManager`. Manages tokenization of the raw code
    and query strings on instantiation. The tokenization is padded and truncated to
    a max length as defined in the training.yml config file.
    """
    def __init__(
            self,
            model_name: str,
            code_lang: str,
            query_langs: Optional[List[str]]=None,
            training: bool=True,
            tiny: bool=False
        ) -> None:
        """
        Initiate a CodeSearchNet dataset for training or evaluation

        Args:
            model_name (str): The name of the model being trained, e.g. CodeBERT or NBOW.
                Controls which type of tokenization is applied.
            code_lang (str): The programming language to download or load from disk.
            query_langs ([str], Optional): Specifies which natural languages to filter
                the queries on. Defaults to `None` which means no filtering is done.
            training (bool, Optional): Whether to use the training, testing and validation data or
                the final evaluation data. Defaults to `True` (using the training, testing and validation data)
            tiny (bool, Optional): Only load a small subset of data. Defaults to `False`. 
        """
        super().__init__()
        model_name = model_name.upper()
        self._tokenizer_manager = CSNetTokenizerManager(model_name, code_lang, tiny, query_langs)
        self._model_name = model_name
        self._code_lang = code_lang
        self._query_langs = query_langs
        self._training = training
        self._data_manager = CSNetDataManager(code_lang, query_langs, tiny=tiny)
        logger.debug(
            "model_name=%s code_lang=%s query_langs=%s training=%s" 
            % (model_name, code_lang, query_langs, training)
        )
        tokenized_data = []
        tokenized_data.append(self._tokenized(
            self._tokenizer_manager.get_tokenizer(TRAINING.SEQUENCE_LENGTHS.CODE),
            "code",
            self._code_cache_file
        ))
        if self._training:
            tokenized_data.append(self._tokenized(
                self._tokenizer_manager.get_tokenizer(TRAINING.SEQUENCE_LENGTHS.QUERY),
                "query",
                self._query_cache_file
            ))
        self._tensor_data = TensorDataset(*tokenized_data)

    def _generic_cache_file(self, name: str, eval_cache=False) -> str:
        """
        Returns the string path to a generic named model specific cache file
        """
        size = "tiny" if self._data_manager._tiny else "full"
        file = f"{size}_{name}"
        if eval_cache:  # E.g. for the evaluation tokenized cache files
            return get_lang_dir(self._code_lang) / Path(file)
        return get_model_dir(self._code_lang, self._query_langs) / Path(file)

    @property
    def _bpe_cache_file(self) -> Path:
        """
        Path to where the custom BPE tokenizer should be serialized to and from
        """
        return self._generic_cache_file("bpe_tokenizer.json")

    @property
    def _code_cache_file(self) -> Path:
        """
        Path to where pre-tokenized queries should be serialized to and from
        """
        name = f"{self._model_name}_code_tokens.pkl"
        return self._generic_cache_file(name, eval_cache=not self._training)

    @property
    def _query_cache_file(self) -> Path:
        """
        Path to where pre-tokenized queries should be serialized to and from
        """
        name = f"{self._model_name}_query_tokens.pkl"
        return self._generic_cache_file(name, eval_cache=not self._training)

    @property
    def _source_data(self) -> pd.DataFrame:
        """
        Grabs the appropriate Pandas `DataFrame` from the data manager when required
        """
        logger.info("Setting up CodeSearchNet dataset for %s" % ("training" if self._training else "evaluation"))
        source = self._data_manager.corpus if self._training else self._data_manager.eval
        logger.debug("Found %d rows of data" % len(source))
        return source

    def _get_tokenizer(self, sequence_length: int) -> TokenizerFunction:
        """
        Get a tokenizer function mapping strings to `torch.Tensor`s for the
        current dataset. Will either use a pretrained BERT-like tokenizer or
        train a BPE tokenizer from scratch on the data, depending on the model
        name given on initialization. 
        """
        if self._model_name in MODELS:
            logger.debug("Use pretrained tokenizer for %s" % self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(MODELS[self._model_name])
            return partial(
                tokenizer.encode,
                max_length=sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        logger.debug("No configured tokenizer found for %s. Using a BPE tokenizer" % self._model_name)
        return self._bpe_tokenizer(sequence_length)

    def _bpe_tokenizer(self, max_length: int) -> TokenizerFunction:
        """
        Train or load a previously trained BPE tokenizer which will pad or truncate the data
        to a given `max_length`. The tokenizer will be trained on both codes and queries.
        """
        if self._bpe_cache_file.exists():
            logger.debug("Found an existing BPE tokenizer at %s" % self._bpe_cache_file)
            tokenizer = Tokenizer.from_file(str(self._bpe_cache_file))
        else:
            logger.info("Training a new BPE tokenizer")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            # This is really silly, but apparently the saved JSON becomes corrput if there is no
            # pre-tokenizer. This is why we "merge" the pretokenized data in preprocessing.py
            # in order to have a common input data shape for both BPE and BERT-like tokenizers 
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=TRAINING.VOCABULARY.SIZE,
                min_frequency=TRAINING.VOCABULARY.INCLUDE_THRESHOLD,
                special_tokens=["[PAD]", "[UNK]"]
            )
            # Train on the corpus which has possibly been filtered on natural language
            data_pairs = self._data_manager.corpus[["code", "query"]].values.tolist()
            # Concatenate query and code tokens in each row to one sequence
            data_stream = chain.from_iterable(data_pairs)
            # Train on the entire sequence of queries and codes,
            # and serialize the resulting tokenizer to JSON
            tokenizer.train_from_iterator(data_stream, trainer)
            tokenizer.save(str(self._bpe_cache_file))
            logger.info("Saved BPE tokenizer to %s" % self._bpe_cache_file)
        
        logger.debug("BPE tokenizer has vocabulary size %d" % tokenizer.get_vocab_size())

        # Set up padding and truncation to match the BERT-like tokenizers
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=max_length
        )
        tokenizer.enable_truncation(max_length=max_length)
        # Return a callable with the same interface as for the BERT-like tokenizers
        return lambda example: torch.tensor(tokenizer.encode(example).ids)

    def _tokenized(
            self,
            tokenize: TokenizerFunction,
            field: str,
            cache_file: Path
        ) -> torch.Tensor:
        """
        Applies a given `TokenizerFunction` to a sequence of strings. The resuls are stacked
        into one tensor of shape (N, L) where N is the number of data samples and L is the
        maximum length of the model. The data is loaded from the data manager in a lazy way, i.e.
        it will only load text data if there is no pre-tokenized cache file

        The results can be serialized to and from a given `cache_file` to save time on subsequent
        training runs
        """ 
        # Check if there are cached features first
        if cache_file.exists():
            logger.debug("Found pretokenized data at %s" % cache_file)
            return torch.load(cache_file)
        
        if not self._data_manager._tiny:
            logger.warning("Tokenizing full data set from scratch. This might take a while")

        # Load in text data from the data manager
        data = self._source_data[field]
        # Process and save features
        tokens = [tokenize(sample) for sample in tqdm(data, desc="Tokenizing", unit="samples")]
        tokens = torch.stack(tokens, dim=0)
        torch.save(tokens, cache_file)
        logger.info("Saved tokenized data to %s" % cache_file)
        return tokens

    def __len__(self) -> Iterator:
        """
        The number of samples in the dataset. Needs to be here for the PyTorch Dataset to be valid.
        """
        return len(self._tensor_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a code-query pair from the internal `TensorDataset` as an appropriately named dictionary.
        """
        sample = self._tensor_data[idx]
        return dict(zip(("code", "query"), sample))


class CSNetTokenizerManager:
    """
    Handles the various kinds of tokenizers requried by the CodeSearchNet dataset
    """
    def __init__(
            self,
            model_name: str,
            code_lang: str,
            tiny: bool,
            query_langs: Optional[List[str]]
        ) -> None:
        self._model_name = model_name.upper()
        self._code_lang = code_lang
        self._tiny = tiny
        self._query_langs = query_langs

    def _generic_cache_file(self, name: str, eval_cache=False) -> str:
        """
        Returns the string path to a generic named model specific cache file
        """
        size = "tiny" if self._tiny else "full"
        file = f"{size}_{name}"
        if eval_cache:  # E.g. for the evaluation tokenized cache files
            return get_lang_dir(self._code_lang) / Path(file)
        return get_model_dir(self._code_lang, self._query_langs) / Path(file)

    @property
    def _bpe_cache_file(self) -> Path:
        """
        Path to where the custom BPE tokenizer should be serialized to and from
        """
        return self._generic_cache_file("bpe_tokenizer.json")

    def get_tokenizer(self, sequence_length: int) -> TokenizerFunction:
        """
        Get a tokenizer function mapping strings to `torch.Tensor`s for the
        current dataset. Will either use a pretrained BERT-like tokenizer or
        train a BPE tokenizer from scratch on the data, depending on the model
        name given on initialization. 
        """
        if self._model_name in MODELS:
            logger.debug("Use pretrained tokenizer for %s" % self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(MODELS[self._model_name])
            return partial(
                tokenizer.encode,
                max_length=sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        logger.debug("No configured tokenizer found for %s. Using a BPE tokenizer" % self._model_name)
        return self._bpe_tokenizer(sequence_length)

    def _bpe_tokenizer(self, max_length: int) -> TokenizerFunction:
        """
        Train or load a previously trained BPE tokenizer which will pad or truncate the data
        to a given `max_length`. The tokenizer will be trained on both codes and queries.
        """
        if self._bpe_cache_file.exists():
            logger.debug("Found an existing BPE tokenizer at %s" % self._bpe_cache_file)
            tokenizer = Tokenizer.from_file(str(self._bpe_cache_file))
        else:
            logger.info("Training a new BPE tokenizer")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            # This is really silly, but apparently the saved JSON becomes corrput if there is no
            # pre-tokenizer. This is why we "merge" the pretokenized data in preprocessing.py
            # in order to have a common input data shape for both BPE and BERT-like tokenizers 
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=TRAINING.VOCABULARY.SIZE,
                min_frequency=TRAINING.VOCABULARY.INCLUDE_THRESHOLD,
                special_tokens=["[PAD]", "[UNK]"]
            )
            # Train on the corpus which has possibly been filtered on natural language
            data_pairs = self._data_manager.corpus[["code", "query"]].values.tolist()
            # Concatenate query and code tokens in each row to one sequence
            data_stream = chain.from_iterable(data_pairs)
            # Train on the entire sequence of queries and codes,
            # and serialize the resulting tokenizer to JSON
            tokenizer.train_from_iterator(data_stream, trainer)
            tokenizer.save(str(self._bpe_cache_file))
            logger.info("Saved BPE tokenizer to %s" % self._bpe_cache_file)
        
        logger.debug("BPE tokenizer has vocabulary size %d" % tokenizer.get_vocab_size())

        # Set up padding and truncation to match the BERT-like tokenizers
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=max_length
        )
        tokenizer.enable_truncation(max_length=max_length)
        # Return a callable with the same interface as for the BERT-like tokenizers
        return lambda example: torch.tensor(tokenizer.encode(example).ids)


class CSNetDataModule(pl.LightningDataModule):
    """
    Responsible for preparing data splits for training and providing data for evaluation.
    This is the main class instantiated during training.
    """
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        """
        Add data module specific arguments to a parent `ArgumentParser`
        """
        parser = parent_parser.add_argument_group("CSNetDataModule")
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--code_lang", type=str, choices=DATA.AVAILABLE_LANGUAGES)
        parser.add_argument("--query_langs", type=List[str], required=False, default=None)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--tiny", action="store_true")
        return parent_parser

    def __init__(self, hparams: Namespace) -> None:
        """
        Initializes a CodeSearchNet data module for training, validation, testing and final evaluation.

        Hyperparameters:
            batch_size (int): The number of samples in each mini-batch
            num_workers (int): The number of workers to use in the data loaders
            encoder_type (Encoder.Types): The name of the model being trained, e.g. CodeBERT or NBOW.
                Controls which type of tokenization is applied.
            code_lang (str): The programming language to download or load from disk.
            query_langs ([str], Optional): Specifies which natural languages to filter
                the queries on. Defaults to `None` which means no filtering is done.
            training (bool, Optional): Whether to use the training, testing and validation data or
                the final evaluation data. Defaults to `True` (using the training, testing and validation data)
            tiny (bool, Optional): Only load a small subset of data. Defaults to `False`. 
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self._model_name = self.hparams.encoder_type.value.upper()
        # Internal member fields for holding data splits
        self._train_split = None
        self._valid_split = None
        self._test_split = None
        self._eval_data = None

    def prepare_data(self) -> None:
        """
        Takes care of downloading and processing the data, saving the results to disk for later.
        """
        CSNetDataset(
            self._model_name,
            self.hparams.code_lang,
            self.hparams.query_langs,
            tiny=self.hparams.tiny
        )
        CSNetDataset(
            self._model_name,
            self.hparams.code_lang,
            self.hparams.query_langs,
            tiny=self.hparams.tiny,
            training=False
        )
        
    def setup(self, stage: Optional[str]=None) -> None:
        """
        Applies tokenization and train-valid-test splits to the data when called with stage set to
        "fit", "validate" or "test". The splits are controlled in the training.yml config file.
        When called with stage set to "predict", the evaluation data is returned. This evaluation
        data is intended for the final NDGC relevance evaluation over the whole dataset.
        """
        dataset = CSNetDataset(
            self._model_name,
            self.hparams.code_lang,
            self.hparams.query_langs,
            training=True,
            tiny=self.hparams.tiny
        )
        if stage in ("fit", "validate", "test", None):
            # Define the split sizes
            N = len(dataset)
            splits = [int(split * N) for split in TRAINING.DATA_SPLITS]
            diff = N - sum(splits)
            splits[0] += diff
            # Do the random splits and tokenize data
            self._train_split, self._valid_split, self._test_split = random_split(dataset, splits)
            
        if stage in ("predict", None):
            with open(DATA.RELEVANCE.QUERIES, "rt") as source:
                queries = [query.strip() for query in source]
            self._eval_data = queries
    
    def train_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for model training.
        """
        return DataLoader(
            self._train_split,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
            
    def val_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for model validation.
        """
        return DataLoader(
            self._valid_split,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for model testing. Uses fixed batches as per the
        original CodeSearchNet paper.
        """        
        return DataLoader(
            self._test_split,
            batch_size=TRAINING.MRR_DISTRACTORS + 1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for NDCG evaluation.
        """    
        return DataLoader(
            self._eval_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
