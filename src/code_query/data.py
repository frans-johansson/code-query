"""
Classes for downloading and processing CodeSearchNet (CSNet) data for training.
Most of the configuration options for these procedures can be controlled in the
data.yml, training.yml and models.yml configuration files.
"""
import zipfile
from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Iterator, List, Optional, Sequence
from pathlib import Path
from functools import cached_property, partial
from itertools import islice, chain

import torch
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
        # Check if the raw data for the given language is available
        if not CSNetDataManager.has_downloaded(code_lang):
            logger.info("Did not find data for %s. Downloading now." % code_lang)
            CSNetDataManager.download_raw(code_lang)
        else:
            logger.info("Found downloaded data for %s" % code_lang)
        if not CSNetDataManager.has_processed(code_lang):
            logger.info("Did not find processed data for %s. Processing now." % code_lang)
            CSNetDataManager.process_raw(code_lang, query_langs)
        else:
            logger.info("Found processed data for %s" % code_lang)
    
    @cached_property
    def corpus(self) -> pd.DataFrame:
        """
        Provides the concatation of the processed raw training, testing and validation data 
        """
        assert CSNetDataManager.has_processed(self._code_lang), "Could not find processed language data"
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
        self._model_name = model_name
        self._code_lang = code_lang
        self._query_langs = query_langs
        self._data_manager = CSNetDataManager(code_lang, query_langs, tiny=tiny)
        self._source_data = self._data_manager.corpus if training else self._data_manager.eval
        logger.info("Setting up CodeSearchNet dataset")
        logger.debug(
            "model_name=%s code_lang=%s query_langs=%s training=%s" 
            % (model_name, code_lang, query_langs, training)
        )
        logger.debug("Found %d rows of data" % len(self._source_data))
        logger.info("Tokenizing code data")
        code_tokenized = self._tokenized(
            self._get_tokenizer(TRAINING.SEQUENCE_LENGTHS.CODE),
            self._source_data["code"].values.tolist(),
            self._code_cache_file
        )
        logger.info("Tokenizing query data")
        query_tokenized = self._tokenized(
            self._get_tokenizer(TRAINING.SEQUENCE_LENGTHS.QUERY),
            self._source_data["query"].values.tolist(),
            self._query_cache_file
        )
        self._code_and_query = TensorDataset(code_tokenized, query_tokenized)
        logger.info("Done setting up CodeSearchNet dataset")

    def _generic_cache_file(self, name: str) -> str:
        """
        Returns the string path to a generic named model specific cache file
        """
        size = "tiny" if self._data_manager._tiny else "full"
        file = f"{size}_{name}"
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
        return self._generic_cache_file(name)

    @property
    def _query_cache_file(self) -> Path:
        """
        Path to where pre-tokenized queries should be serialized to and from
        """
        name = f"{self._model_name}_query_tokens.pkl"
        return self._generic_cache_file(name)

    def _get_tokenizer(self, sequence_length: int) -> TokenizerFunction:
        """
        Get a tokenizer function mapping strings to `torch.Tensor`s for the
        current dataset. Will either use a pretrained BERT-like tokenizer or
        train a BPE tokenizer from scratch on the data, depending on the model
        name given on initialization. 
        """
        if self._model_name in MODELS:
            logger.info("Use pretrained tokenizer for %s" % self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(MODELS[self._model_name])
            return partial(
                tokenizer.encode,
                max_length=sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        logger.info("No pretrained tokenizer found for %s. Using a BPE tokenizer" % self._model_name)
        return self._bpe_tokenizer(sequence_length)

    def _bpe_tokenizer(self, max_length: int) -> TokenizerFunction:
        """
        Train or load a previously trained BPE tokenizer which will pad or truncate the data
        to a given `max_length`. The tokenizer will be trained on both codes and queries.
        """
        if self._bpe_cache_file.exists():
            logger.info("Found an existing BPE tokenizer at %s" % self._bpe_cache_file)
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
            data: Sequence,
            cache_file: Path
        ) -> torch.Tensor:
        """
        Applies a given `TokenizerFunction` to a sequence of strings. The resuls are stacked
        into one tensor of shape (N, L) where N is the number of data samples and L is the
        maximum length of the model.

        The results can be serialized to and from a given `cache_file` to save time on subsequent
        training runs
        """ 
        # Check if there are cached features first
        if cache_file.exists():
            logger.info("Found pretokenized data at %s" % cache_file)
            return torch.load(cache_file)
        
        if not self._data_manager._tiny:
            logger.warning("Tokenizing full data set from scratch. This might take a while")

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
        return len(self._source_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a code-query pair from the internal `TensorDataset` as an appropriately named dictionary.
        """
        code, query = self._code_and_query[idx] 
        return {
            "code": code,
            "query": query
        }


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
            encoder_type (str): The name of the model being trained, e.g. CodeBERT or NBOW.
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
        self._model_name = hparams.encoder_type.value.upper()
        # Internal member fields for holding data splits
        self._train_split = None
        self._valid_split = None
        self._test_split = None
    
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
        
    def setup(self, stage: Optional[str]=None) -> None:
        """
        Applies tokenization and train-valid-test splits to the data when called with stage set to
        "fit", "validate" or "test". The splits are controlled in the training.yml config file.
        When called with stage set to "predict", the evaluation data is returned. This evaluation
        data is intended for the final NDGC relevance evaluation over the whole dataset.
        """
        if stage in ("fit", "validate", "test", None):
            corpus_dataset = CSNetDataset(
                self._model_name,
                self.hparams.code_lang,
                self.hparams.query_langs,
                training=True,
                tiny=self.hparams.tiny
            )
            # Define the split sizes
            N = len(corpus_dataset)
            splits = [int(split * N) for split in TRAINING.DATA_SPLITS]
            diff = N - sum(splits)
            splits[0] += diff
            # Do the random splits and tokenize data
            self._train_split, self._valid_split, self._test_split = random_split(corpus_dataset, splits)
            
        if stage in ("predict", None):
            # TODO: Set up prediction data for NDCG
            # eval_dataset = CSNetDataset(self.model_name, self.code_lang, self.query_langs, training=False)
            raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for model training.
        """
        return DataLoader(
            self._train_split,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )
            
    def val_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for model validation.
        """
        return DataLoader(
            self._valid_split,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for model testing. Uses fixed batches as per the
        original CodeSearchNet paper.
        """        
        return DataLoader(
            self._test_split,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch `DataLoader` for NDCG evaluation.
        """    
        raise NotImplementedError()
