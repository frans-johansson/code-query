"""
Defines a Pytorch-Lightning module for setting up and training the CodeQuery model
"""
from argparse import ArgumentParser, Namespace
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from torchmetrics import RetrievalMRR

from code_query.model.encoder import Encoder


class CodeQuery(pl.LightningModule):
    """
    Represents the main `CodeQuery` model consisting of a pair of siamese encoder
    networks which map codes and queries to a common latent space. The model then
    trained using a simple triplet loss where positive code-query pairs are mapped
    closer together in the latent space, and negative pairs are mapped further apart
    """
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        """
        Add model specific arguments to a parent `ArgumentParser`
        """
        # Handle model args
        parser = parent_parser.add_argument_group("CodeQuery")
        parser.add_argument("--encoder_type", type=Encoder.Types)
        parser.add_argument("--learning_rate", type=float, default=0.1)
        # Set up Encoder args
        try:
            temp_args, _ = parent_parser.parse_known_args()
            parent_parser = Encoder.get_type(temp_args.encoder_type).add_argparse_args(parent_parser)
        except:
            raise ValueError("Please provide a value for the --encoder_type argument")
        return parent_parser

    def __init__(self, hparams: Namespace) -> None:
        """
        Set up a `CodeQuery` model for training

        Args:
            learning_rate (float): The initial learning rate for the
                AdamW optimizer. Defaults to 0.1.
            encoder_type (Encoder.Types): Name of the encoder type to use,
                e.g. "nbow" or "bert".
            encoder_args (kwargs): Additional keyword arguments required by
                the encoder module.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        EncoderClass = Encoder.get_type(hparams.encoder_type)
        self.encoder = EncoderClass(hparams)
        # For test loss
        self.mrr = RetrievalMRR()

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

    def _cosine_sim_mat(self, encoded_queries, encoded_codes) -> torch.Tensor:
        """
        Computes the cosine similarity matrix for two sets of encoded codes and queries,
        such that the values at position (i, j) will be the cosine similarity of the ith
        query compared to the jth code snippet
        """
        norm_encoded_queries = encoded_queries / encoded_queries.norm(dim=1, keepdim=True)
        norm_encoded_codes = encoded_codes / encoded_codes.norm(dim=1, keepdim=True)
        mat = norm_encoded_queries @ norm_encoded_codes.T  # (queries, codes)
        return mat

    def _training_loss(self, encoded_codes, encoded_queries) -> torch.Tensor:
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
        mat = encoded_queries @ encoded_codes.T  # (queries, codes)
        n = mat.shape[0]
        preds = mat.view(-1)  # Reshaped on rows, grouped by query
        target = torch.eye(n, dtype=int).view(-1)
        indexes = torch.cat([torch.full(size=(n,), fill_value=i) for i in range(n*idx, n*(idx+1))])
        return (preds, target, indexes)

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

    def on_train_batch_end(self, outputs: torch.Tensor, X: Any, idx: int) -> None:
        if outputs["loss"].isnan():
            raise ValueError(f"Found NaN loss value at batch index {idx}")

    def on_after_backward(self) -> None:
        grads = {name: param.grad for name, param in self.named_parameters()}
        weights = {name: param.data for name, param in self.named_parameters()}
        pass

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
        preds, target, indexes = self._mrr_setup(encoded_codes, encoded_queries)
        self.mrr(preds, target, indexes)
        self.log("test/mrr", self.mrr, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> None:
        """
        Sets up optimizers for the model training process configured with the
        `learning_rate` hyperparameter
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
