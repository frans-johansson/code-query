"""
Defines a Pytorch-Lightning module for setting up and training the CodeQuery model
"""
from abc import ABC, abstractclassmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from torchmetrics import RetrievalMRR
from tqdm import tqdm
from annoy import AnnoyIndex
import pandas as pd

from code_query.model.encoder import Encoder
from code_query.config import TRAINING


# FIXME: Redefining this from data.py to avoid circular imports for now
TokenizerFunction = Callable[[str], torch.Tensor]

class CodeQuery(pl.LightningModule, ABC):
    """Base `CodeQuery` class for both dual and siamese variants"""
    class Types(Enum):
        SIAMESE="siamese"
        DUAL="dual"

    def __init__(self, hparams: Union[Namespace, Dict[str, Any]]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.EncoderClass = Encoder.get_type(self.hparams.encoder_type)
        self.mrr = RetrievalMRR()
        self.index = AnnoyIndex(f=self.hparams.encoding_dim, metric="angular")
        self.eval_lookup = None
        self.tokenizer = None

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        """
        Add model specific arguments to a parent `ArgumentParser`
        """
        # Handle model args
        parser = parent_parser.add_argument_group("CodeQuery")
        parser.add_argument("--model_type", type=CodeQuery.Types, default=CodeQuery.Types.SIAMESE)
        parser.add_argument("--encoder_type", type=Encoder.Types)
        parser.add_argument("--learning_rate", type=float, default=0.1)
        # Set up Encoder args
        try:
            temp_args, _ = parent_parser.parse_known_args()
            parent_parser = Encoder.get_type(temp_args.encoder_type).add_argparse_args(parent_parser)
        except:
            raise ValueError("Please provide a value for the --encoder_type argument")
        return parent_parser

    @staticmethod
    def get_type(type: Types):
        """
        Returns a class type corresponding to the given type string or enum
        """
        if type == CodeQuery.Types.SIAMESE:
            return CodeQuerySiamese
        if type == CodeQuery.Types.DUAL:
            return CodeQueryDual
        raise NotImplementedError()

    @abstractclassmethod
    def _encode_pair(self, X: Any) -> Tuple[torch.Tensor]:
        """
        Encodes a data input of codes and queries and returns a tuple of the results.
        The inputs are expected to be a dictionary of batches with keys "code" and
        "query" for code and query samples respectively.
        """
        pass

    def training_step(self, X: Any) -> torch.Tensor:
        """
        Performs a single trainig step and returns the loss

        Args:
            X ({"code": Tensor, "query": Tensor}): A dictionary of the tokenized
                code and query input tensors of shape (B, L) where B is the batch size
                and L is the length of each sequence
        """
        encoded_codes, encoded_queries = self._encode_pair(X)
        loss = self._training_loss(encoded_codes, encoded_queries)
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, X: Any, idx: int) -> None:
        """
        Performs a single validation step and returns the loss

        Args:
            X ({"code": Tensor, "query": Tensor}): A dictionary of the tokenized
                code and query input tensors of shape (B, L) where B is the batch size
                and L is the length of each sequence
            idx (int): The index of the current validation data sample
        """
        encoded_codes, encoded_queries = self._encode_pair(X)
        loss = self._training_loss(encoded_codes, encoded_queries)
        preds, target, indexes = self._mrr_setup(encoded_codes, encoded_queries, idx)
        self.mrr(preds, target, indexes)
        self.log("valid/loss", loss, on_step=False, on_epoch=True)
        self.log("valid/mrr", self.mrr, on_step=False, on_epoch=True)

    def test_step(self, X: Any, idx: int) -> None:
        """
        Performs a single test step over a batch, logging the MRR loss over the entire epoch
        """
        encoded_codes, encoded_queries = self._encode_pair(X)
        preds, target, indexes = self._mrr_setup(encoded_codes, encoded_queries, idx)
        loss = self.mrr(preds, target, indexes)
        self.log("test/mrr", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, List[str]]:
        return {
            # TODO: Move n_results to config
            query: self.search(query, n_results=100)
            for query in batch
        }

    def search(self, query: str, n_results: int) -> List[str]:
        """
        Returns search results for a given query. Note that this requires
        the index to be set up with `setup_index` and a tokenizer to be
        prepared with `set_tokenizer`.
        """
        assert self.tokenizer is not None, "Please set the tokenizer with set_tokenizer before running the predictions"
        tokenized_query = self.tokenizer(query)
        encoded_query = self.forward(tokenized_query[None])
        result_idxs = self.index.get_nns_by_vector(encoded_query.squeeze(), n_results)
        return self.eval_lookup.loc[result_idxs, "url"].values.tolist()

    def configure_optimizers(self) -> None:
        """
        Sets up optimizers for the model training process configured with the
        `learning_rate` hyperparameter
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def setup_predictor(
            self,
            eval_lookup: pd.DataFrame,
            tokenizer: TokenizerFunction
        ) -> None:
        """
        Sets the tokenizer and data lookup to use when predicting search query results.
        
        The tokenizer is intended to be applied to search query strings and should be compatible with
        the programming language, natual languages and encoder type used to work properly.
        """
        self.eval_lookup = eval_lookup
        self.tokenizer = tokenizer

    def setup_index(self, data: Iterable, n_trees: int, ann_dir: Path) -> None:
        """
        Indexes the data held by a given iterable which is expected to
        yield dictionaries of (possibly batched) tensors with at least the key
        "code". The indexing is done with the `annoy` indexer, which approximates
        nearest neighbor clustering with a set number of trees which must be specified.
        """
        self.eval()  # Unfortunately this is done outside of the predict-loop, so we don't get automatic eval
        ann_file = ann_dir / "index.ann"
        if ann_file.exists():
            self.index.load(str(ann_file))
            return
        ann_dir.mkdir(exist_ok=True, parents=True)
        self.index.set_seed(TRAINING.SEED)
        for batch_idx, batch in tqdm(enumerate(data), desc="Indexing", total=len(data)):
            with torch.no_grad():  # We also don't get automatic no-grad
                encoded = self.forward(batch["code"]).detach().numpy()
            for idx, sample in enumerate(encoded):
                self.index.add_item(batch_idx + idx, sample)
        self.index.build(n_trees)
        self.index.save(str(ann_file))

    def _cosine_sim_mat(self, encoded_queries: torch.Tensor, encoded_codes: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity matrix for two sets of encoded codes and queries,
        such that the values at position (i, j) will be the cosine similarity of the ith
        query compared to the jth code snippet
        """
        norm_encoded_queries = encoded_queries / encoded_queries.norm(dim=1, keepdim=True)
        norm_encoded_codes = encoded_codes / encoded_codes.norm(dim=1, keepdim=True)
        mat = norm_encoded_queries @ norm_encoded_codes.T  # (queries, codes)
        return mat

    def _training_loss(self, encoded_codes: torch.Tensor, encoded_queries: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the training and validation loss for an encoded pair
        """
        mat = self._cosine_sim_mat(encoded_queries, encoded_codes)
        n = mat.shape[0]
        off = mat.masked_select(~torch.eye(n, dtype=bool, device=self.device)).view(n, n - 1)

        pos = 1.0 - mat.diag() # True code-query parings should have low cosine distance
        neg = F.relu(off).max(dim=1).values  # False pairings should have low cosine similarity
        loss = pos + neg
        return loss.mean()

    def _mrr_setup(self, encoded_codes: torch.Tensor, encoded_queries: torch.Tensor, idx: int) -> Tuple[torch.Tensor]:
        """
        Prepares a set of encoded codes and queries for the TorchMetrics MRR loss.
        
        1.The encoded values are converted to a cosine similarity matrix and flattened
            row-wise to a preds tensor.
        2. A target tensor is computed matching up with the diagonal of the cosine similarity
            matrix (i.e. true pairs are coded as targets).
        3. An index tensor is constructed to group the target and preds tensors per query,
            i.e. N contiguous blocks of integers where N is the number of encoded codes and queries.

        Returns a tuple of (preds, target, indexes)  
        """
        mat = self._cosine_sim_mat(encoded_queries, encoded_codes)  # (queries, codes)
        n = mat.shape[0]
        preds = mat.view(-1)  # Reshaped on rows, grouped by query
        target = torch.eye(n, dtype=int).view(-1)  # True where codes and queries match
        # Indexes will look something like [0, 0, ..., 0, 1, 1, ..., 1, ..., N, N, ..., N]
        indexes = torch.cat([torch.full(size=(n,), fill_value=i) for i in range(n*idx, n*(idx+1))])
        return (preds, target, indexes)


class CodeQuerySiamese(CodeQuery):
    """
    Represents the main `CodeQuery` model consisting of a pair of siamese encoder
    networks which map codes and queries to a common latent space. The model then
    trained using a simple triplet-like loss where positive code-query pairs are mapped
    closer together in the latent space, and negative pairs are mapped further apart
    """
    def __init__(self, hparams: Union[Namespace, Dict[str, Any]]) -> None:
        """
        Set up a `CodeQuery` model for training

        Hyperparameters:
            learning_rate (float): The initial learning rate for the
                AdamW optimizer. Defaults to 0.1.
            encoder_type (Encoder.Types): Name of the encoder type to use,
                e.g. "nbow" or "bert".
            encoder_args (kwargs): Additional keyword arguments required by
                the encoder module.
        """
        super().__init__(hparams)
        self.encoder = self.EncoderClass(hparams)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used for inference. Computes the encoding of either
        a tokenized code or query tensor.
        
        Args:
            X: A tokenized code or query sequence 
        """
        return self.encoder(X)

    def _encode_pair(self, X: Any) -> Tuple[torch.Tensor]:
        """
        Encodes a data input of codes and queries and returns a tuple of the results
        """
        codes = X["code"]
        queries = X["query"]
        encoded_codes = self.forward(codes)
        encoded_queries = self.forward(queries)
        return (encoded_codes, encoded_queries)


class CodeQueryDual(CodeQuery):
    """
    Represents the main `CodeQuery` model consisting two separate encoder
    networks which map codes and queries to a common latent space. The model then
    trained using a simple triplet-like loss where positive code-query pairs are mapped
    closer together in the latent space, and negative pairs are mapped further apart
    """
    def __init__(self, hparams: Union[Namespace, Dict[str, Any]]) -> None:
        """
        Set up a `CodeQuery` model for training

        Hyperparameters:
            learning_rate (float): The initial learning rate for the
                AdamW optimizer. Defaults to 0.1.
            encoder_type (Encoder.Types): Name of the encoder type to use,
                e.g. "nbow" or "bert".
            encoder_args (kwargs): Additional keyword arguments required by
                the encoder module.
        """
        super().__init__(hparams)
        self.code_encoder = self.EncoderClass(hparams)
        self.query_encoder = self.EncoderClass(hparams)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used for inference. Computes the encoding of
        a tokenized code tensor specifically.
        
        Args:
            X: A tokenized code sequence 
        """
        return self.code_encoder(X)

    def _encode_pair(self, X: Any) -> Tuple[torch.Tensor]:
        """
        Encodes a data input of codes and queries and returns a tuple of the results
        """
        codes = X["code"]
        queries = X["query"]
        encoded_codes = self.code_encoder(codes)
        encoded_queries = self.query_encoder(queries)
        return (encoded_codes, encoded_queries)
