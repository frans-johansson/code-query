"""
Contains various encoders for code and queries which the CodeQuery model utilizes.
"""
from abc import ABC, abstractstaticmethod
from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Any, Dict, Union

import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig

from code_query.config import MODELS, TRAINING


class Encoder(ABC):
    """
    Base class for encoders. Can also be used to obtain relevant
    classes from a string argument via the internal `Types` enum.
    This class demands that all derived classes implement their own
    static method for argument parsing in order to provide a coherent
    training script API.
    """
    class Types(Enum):
        NBOW = "nbow"
        BERT = "bert"
        ROBERTA = "roberta"
        CODEBERT = "codebert"
        DISTILBERT = "distilbert"

    @staticmethod
    def get_type(encoder_type: Types):
        """
        Returns a class type corresponding to the given type string or enum
        """
        if encoder_type == Encoder.Types.NBOW:
            return NbowEncoder
        else:
            return BertLikeEncoder

    @abstractstaticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Should define all arguments requried to initialize the derived encoder
        class in its own parser group on the parent parser, and return the parent parser
        """
        return parent_parser


class NbowEncoder(pl.LightningModule, Encoder):
    """
    Neural bag-of-words encoder
    """
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add encoder specific arguments to a parent `ArgumentParser`
        """
        parser = parent_parser.add_argument_group("NbowEncoder")
        parser.add_argument("--embedding_dim", type=int, default=128)
        parser.add_argument("--encoding_dim", type=int, default=128)
        parser.add_argument("--encoder_dropout", type=float, default=0.1)
        return parent_parser

    def __init__(self, hparams: Union[Dict[str, Any], Namespace]) -> None:
        """
        Sets up an NBOW encoder for code and queries

        Hyperparameters:
            embedding_dim (int): Size of the embedding dimensions.
                Defaults to 128.
            encoding_dim (int): Size of the hidden dimensions for each sequence.
                Defaults to 128.
            encoder_dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.embed = nn.Embedding(
            num_embeddings=TRAINING.VOCABULARY.SIZE,
            embedding_dim=self.hparams.embedding_dim,
            padding_idx=0,
            scale_grad_by_freq=True
        )
        self.fc = nn.Linear(
            in_features=self.hparams.embedding_dim,
            out_features=self.hparams.encoding_dim
        )
        self.bn = nn.BatchNorm1d(self.hparams.encoding_dim)
        self.drop = nn.Dropout(p=self.hparams.encoder_dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass through the encoder, producing a latent representation of the input

        Args:
            X (Tensor): A tensor of shape (B, L) representing the input sequence
                where B is the batch size and L is the sequence length of each sample

        Returns: A tensor of shape (B, H) of latent representations, where H is set by the
            `encoding_dim` hyperparameter
        """
        embeddings = self.embed(X)  # (batch, seq, embedding)
        nbow = torch.mean(embeddings, dim=1)  # (batch, embedding)
        hidden = self.fc(nbow)
        hidden = self.bn(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.drop(hidden)
        return hidden


class BertLikeEncoder(pl.LightningModule, Encoder):
    """
    An encoder using a BERT-like embedding model
    """
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add encoder specific arguments to a parent `ArgumentParser`
        """
        parser = parent_parser.add_argument_group("NbowEncoder")
        parser.add_argument("--encoding_dim", type=int, default=128)
        parser.add_argument("--encoder_dropout", type=float, default=0.1)
        return parent_parser

    def __init__(self, hparams: Union[Dict[str, Any], Namespace]) -> None:
        """
        Sets up a BERT-like encoder for code and queries

        Hyperparameters:
            encoder_type (Encoder.Types): A string specifying the BERT variant to use.
            encoding_dim (int): Size of the hidden dimensions for each sequence.
                Defaults to 128.
            encoder_dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        model_name = MODELS[self.hparams.encoder_type.value.upper()]
        self.embed = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(
            in_features=AutoConfig.from_pretrained(model_name).hidden_size,
            out_features=self.hparams.encoding_dim
        )
        self.bn = nn.BatchNorm1d(self.hparams.encoding_dim)
        self.drop = nn.Dropout(p=self.hparams.encoder_dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass through the encoder, producing a latent representation of the input

        Args:
            X (Tensor): A tensor of shape (B, 1, L) representing the input sequence
                where B is the batch size and L is the sequence length of each sample

        Returns: A tensor of shape (B, H) of latent representations, where H is set by the
            `encoding_dim` hyperparameter
        """
        embeddings = self.embed(X.squeeze(1)).last_hidden_state  # (batch, seq, embedding)
        nbow = torch.mean(embeddings, dim=1)  # (batch, embedding)
        hidden = self.fc(nbow)
        hidden = self.bn(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.drop(hidden)
        return hidden
