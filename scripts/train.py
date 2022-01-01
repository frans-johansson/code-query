"""
Script for training CodeQuery modlels on the CodeSearchNet data
provided by the `code_query.data` submodule 
"""
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from code_query.data import CSNetDataModule
from code_query.model import CodeQuery
from code_query.config import WANDB, TRAINING
from code_query.utils.helpers import get_ckpt_dir
from code_query.utils.logging import get_run_name


if __name__ == "__main__":
    # Take care of the seed values
    pl.seed_everything(TRAINING.SEED)

    # Handle command line arguments
    parser = ArgumentParser("CodeQuery training")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser = CodeQuery.add_argparse_args(parser)
    parser = CSNetDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    # Set up model and data from parsed hyperparameters
    model = CodeQuery(hparams)
    data_module = CSNetDataModule(hparams)

    # Weights and Biases logger
    logger = None
    if hparams.wandb:
        logger = WandbLogger(
            name=get_run_name(hparams.encoder_type),
            project=WANDB.PROJECT_NAME,
            save_dir=Path(WANDB.DIR),
            log_model="all"
        )
        logger.watch(model, log="all")

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        dirpath=get_ckpt_dir(
            hparams.encoder_type,
            hparams.code_lang,
            hparams.query_langs
        ),
        monitor="valid/loss",
        mode="min"
    )

    # Early stopping
    stop_callback = EarlyStopping(
        monitor="valid/loss",
        min_delta=TRAINING.EARLY_STOPPING.MIN_DELTA,
        patience=TRAINING.EARLY_STOPPING.PATIENCE
    )

    # Set up and run training
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        callbacks=[ckpt_callback, stop_callback]
    )
    trainer.fit(model, data_module)
