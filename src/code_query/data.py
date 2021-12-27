"""
Classes for downloading and processing CodeSearchNet (CSNet) data for training.

- `CSNetDataManager`: Handles downloading, processing and caching the raw CSNet data. Provides
    iterators over the full training and evaluation corpus via class properties.
- `CSNetDataset`: Implements the PyTorch IterableDataset over either the training or evaluation
    corpus supplied by the `CSNetDataManager`
- `CSNetDataModule`: Responsible for preparing data splits and tokenizing the raw text data
    for training.
"""
import zipfile
from typing import Iterator, List, Optional, Sequence
from pathlib import Path
from functools import cached_property, partial
from itertools import islice

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset, random_split
from transformers import AutoTokenizer
from tqdm import tqdm

from code_query.config import DATA, MODELS, TRAINING
from code_query.utils.helpers import download_file, get_lang_dir, get_model_dir
from code_query.utils.logging import get_logger
from code_query.utils.serialize import jsonl_gzip_load
from code_query.utils.preprocessing import process_evaluation_data, process_training_data


logger = get_logger("data")


class CSNetDataManager(object):
    def __init__(
            self, 
            code_lang: str,
            query_langs: Optional[List[str]]=None,
            tiny: bool=False
        ) -> None:
        super().__init__()
        assert code_lang in DATA.AVAILABLE_LANGUAGES, "Unknown programming language"
        # Root directory of cached files
        self._model_dir = get_model_dir(code_lang, query_langs)
        self._root_dir = get_lang_dir(code_lang)
        self._code_lang = code_lang
        self._tiny = tiny
        # Check if the raw data for the given language is available
        if not CSNetDataManager.has_downloaded(code_lang):
            logger.info("Did not find data for {%s}. Downloading now." % code_lang)
            CSNetDataManager.download_raw(code_lang)
            CSNetDataManager.process_raw(code_lang, query_langs)
    
    @cached_property
    def corpus(self) -> pd.DataFrame:
        assert CSNetDataManager.has_processed(self._code_lang), "Could not find processed language data"
        logger.info("Reading corpus data")
        return self._load_as_dataframe(self._model_dir / "corpus.jsonl.gz")

    @cached_property
    def eval(self) -> pd.DataFrame:
        logger.info("Reading eval data")
        return self._load_as_dataframe(self._root_dir / "eval.jsonl.gz")

    def _load_as_dataframe(self, path) -> pd.DataFrame:
        data_stream = jsonl_gzip_load(path)
        if self._tiny:
            data_stream = islice(data_stream, DATA.TINY_SIZE)
        return pd.DataFrame.from_records(data_stream)

    @staticmethod
    def has_downloaded(code_lang: str) -> bool:
        dir = Path(DATA.DIR.RAW) / code_lang
        return dir.exists()

    @staticmethod
    def has_processed(code_lang: str) -> bool:
        dir = get_lang_dir(code_lang)
        return dir.exists()

    @staticmethod
    def process_raw(code_lang: str, query_langs: Optional[List[str]]=None) -> None:
        assert code_lang in DATA.AVAILABLE_LANGUAGES, "Unknown programming language"
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
        assert code_lang in DATA.AVAILABLE_LANGUAGES, "Unknown programming language"
        assert not CSNetDataManager.has_downloaded(code_lang), "Programming language already downloaded"
        raw_path = Path(DATA.DIR.RAW)
        raw_path.mkdir(exist_ok=True)
        url = DATA.URL.format(language=code_lang)
        zip_path = raw_path / f"{code_lang}.zip"
        # Download
        download_file(url, zip_path, description=f"Downloading {code_lang}.zip")
        # Unzip
        logger.info("Unzipping {%s}" % zip_path)
        with zipfile.ZipFile(zip_path, "r") as source:
            source.extractall(DATA.DIR.RAW)
        logger.info("Unzip done")
        # Clean up
        zip_path.unlink()


class CSNetDataset(Dataset):
    def __init__(self, model_name: str, code_lang: str, query_langs: Optional[List[str]]=None, training: bool=True, tiny: bool=False) -> None:
        super().__init__()
        model_name = model_name.upper()
        data_manager = CSNetDataManager(code_lang, query_langs, tiny=tiny)
        self.training = training
        self._source_data = data_manager.corpus if training else data_manager.eval
        logger.info(
            "Setting up CodeSearchNet dataset: model_name=%s code_lang=%s query_langs=%s training=%s" 
            % (model_name, code_lang, query_langs, training)
        )
        logger.info("Found %d rows of data" % len(self._source_data))
        logger.info("Tokenizing code data")
        code_tokenized = CSNetDataset.tokenized(
            model_name,
            self._source_data["code_tokens"].values.tolist(),
            TRAINING.SEQUENCE_LENGTHS.CODE
        )
        logger.info("Tokenizing query data")
        query_tokenized = CSNetDataset.tokenized(
            model_name,
            self._source_data["query_tokens"].values.tolist(),
            TRAINING.SEQUENCE_LENGTHS.QUERY
        )
        self._code_and_query = TensorDataset(code_tokenized, query_tokenized)
        logger.info("Done setting up CodeSearchNet dataset")

    @staticmethod
    def tokenized(model_name: str, data: Sequence, sequence_length: int) -> torch.tensor:
        # Get the appropriate tokenizer
        if model_name in MODELS:
            tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
            logger.info("Using pretrained tokenizer: %s" % tokenizer.__str__())
        else:
            # TODO: Train a custom tokenizer
            raise NotImplementedError()
        
        # Process and save features
        # input_ids = []
        # attention_masks = []
        # for sample in tqdm(data, desc="Tokenizing", unit="samples"):
        #     tokenizer_output = tokenizer(
        #         sample,
        #         padding="max_length",
        #         truncation=True,
        #         is_split_into_words=True,
        #         return_tensors="pt"
        #     )
        #     input_ids.append(tokenizer_output.input_ids)
        #     attention_masks.append(tokenizer_output.attention_mask)

        # return TensorDataset(torch.cat(input_ids), torch.cat(attention_masks))
        tokenize = partial(
            tokenizer.encode,
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        tokens = [tokenize(sample) for sample in tqdm(data, desc="Tokenizing", unit="samples")]
        return torch.cat(tokens)

    def __len__(self) -> Iterator:
        return len(self._source_data)
    
    def __getitem__(self, idx):
        code, query = self._code_and_query[idx] 
        return {
            "code": code,
            "query": query
        }


class CSNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            code_lang: str,
            batch_size: int,
            query_langs: Optional[List[str]]=None,
            tiny: bool=False
        ) -> None:
        super().__init__()
        self.code_lang = code_lang
        self.query_langs = query_langs
        self.model_name = model_name.upper()
        self.batch_size = batch_size
        self.tiny = tiny
        self._train_split = None
        self._valid_split = None
        self._test_split = None
    
    def prepare_data(self):
        CSNetDataset(self.model_name, self.code_lang, self.query_langs)
        
    def setup(self, stage: Optional[str]=None):
        if stage in ("fit", "validate", "test", None):
            corpus_dataset = CSNetDataset(self.model_name, self.code_lang, self.query_langs, training=True, tiny=self.tiny)
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
            pass

    def train_dataloader(self):
        return DataLoader(self._train_split, batch_size=self.batch_size, shuffle=True)
            
    def val_dataloader(self):
        return DataLoader(self._valid_split, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self._test_split, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        # TODO: Handle the prediction data for NDGC
        raise NotImplementedError()
