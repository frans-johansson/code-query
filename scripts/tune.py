"""
Script for running hyperparameter tuning of CodeQuery models using Weights and Biases sweeps
"""
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from code_query.data import CSNetDataModule
from code_query.model import CodeQuery
from code_query.config import WANDB, TRAINING


if __name__ == "__main__":
    # Take care of the seed values
    pl.seed_everything(TRAINING.SEED)

    # Handle command line arguments
    parser = ArgumentParser("CodeQuery tuning")
    parser = CodeQuery.add_argparse_args(parser)
    parser = CSNetDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    # Set up model and data from parsed hyperparameters
    model = CodeQuery(hparams)
    data_module = CSNetDataModule(hparams)

    wandb_dir = Path(WANDB.DIR)
    wandb_dir.mkdir(exist_ok=True, parents=True)
    logger = WandbLogger(
        project=WANDB.PROJECT_NAME,
        save_dir=wandb_dir,
        log_model=False,
        tags=["Sweep", hparams.encoder_type.value]
    )
    logger.watch(model, log="gradients", log_graph=False)

    # Set up and run training
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        enable_checkpointing=False
    )
    trainer.fit(model, data_module)
