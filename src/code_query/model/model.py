"""
Defines a Pytorch-Lightning module for setting up and training the CodeQuery model
"""
from argparse import ArgumentParser, Namespace
from typing import Any, Tuple

import torch
from torch import nn
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
        parser.add_argument("--cq_h_dim", type=int, default=128)
        parser.add_argument("--cq_dropout", type=float, default=0.1)
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
            cq_h_dim (int): The dimensions of the final dense latent
                space of the codes and queries. Defaults to 128.
            cq_dropout (float): Dropout rate from the encoder step to
                the final dense representation. Defaults to 0.1.
            encoder_type (Encoder.Types): Name of the encoder type to use,
                e.g. "nbow" or "bert".
            encoder_args (kwargs): Additional keyword arguments required by
                the encoder module.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        EncoderClass = Encoder.get_type(hparams.encoder_type)
        self.encoder = EncoderClass(hparams)
        self.fc = nn.Linear(in_features=hparams.enc_h_dim, out_features=hparams.cq_h_dim)
        self.bn = nn.BatchNorm1d(hparams.cq_h_dim)
        self.drop = nn.Dropout(p=hparams.cq_dropout)
        # For test loss
        self.mrr = RetrievalMRR()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used for inference. Computes the encoding of either
        a tokenized code or query tensor.
        
        Args:
            X: A tokenized code or query sequence 
        """
        encoded = self.encoder(X)
        out = self.fc(encoded)
        out = self.bn(out)
        out = torch.sigmoid(out)
        out = self.drop(out)
        return out

    def _encoded_pair(self, X: Any) -> Tuple[torch.Tensor]:
        """
        Encodes a data input of codes and queries and returns a tuple of the results
        """
        codes = X["code"]
        queries = X["query"]

        h_codes = self.forward(codes)
        h_queries = self.forward(queries)
        return (h_codes, h_queries)

    def _training_loss(self, h_codes, h_queries) -> torch.Tensor:
        """
        Computes and returns the training and validation loss for an encoded pair
        """
        n = self.hparams.batch_size
        mat = h_queries @ h_codes.T  # (queries, codes)
        pos = mat.diag()  # True code-query pairings
        off = mat.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)
        neg = off.exp().sum(dim=1)  # False code-query pairings
        
        loss = torch.log(torch.sum(pos / neg + 10e-9))
        return -loss / n

    def training_step(self, X: Any) -> torch.Tensor:
        """
        Performs a single trainig step and returns the loss

        Args:
            X ({"code": Tensor, "query": Tensor}): A dictionary of the tokenized
                code and query input tensors of shape (B, L) where B is the batch size
                and L is the length of each sequence
        """
        h_codes, h_queries = self._encoded_pair(X)
        loss = self._training_loss(h_codes, h_queries)
        self.log("train/loss", loss)
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
        h_codes, h_queries = self._encoded_pair(X)
        loss = self._training_loss(h_codes, h_queries)
        self.log("valid/loss", loss)
        return loss

    def test_step(self, X: Any, idx: int) -> None:
        """
        Performs a single test step over a batch, logging the MRR loss over the entire epoch
        """
        h_codes, h_queries = self._encoded_pair(X)
        mat = h_queries @ h_codes.T  # (queries, codes)
        n = mat.shape[0]
        preds = mat.view(-1)  # Reshaped on rows, grouped by query
        target = torch.eye(n, dtype=int).view(-1)
        indexes = torch.cat([torch.full(size=(n,), fill_value=i) for i in range(n*idx, n*(idx+1))])
        self.mrr(preds, target, indexes)
        self.log("test/mrr", self.mrr, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> None:
        """
        Sets up optimizers for the model training process configured with the
        `learning_rate` hyperparameter
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
