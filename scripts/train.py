"""
Script for training CodeQuery modlels on the CodeSearchNet data
provided by the `code_query.data` submodule 
"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from code_query.data import CSNetDataModule
from code_query.model import CodeQuery


if __name__ == "__main__":
    # Handle command line arguments
    parser = ArgumentParser("CodeQuery training")
    parser = CodeQuery.add_argparse_args(parser)
    parser = CSNetDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Set up components from parsed hyperparameters
    trainer = Trainer.from_argparse_args(hparams)
    model = CodeQuery(hparams)
    data_module = CSNetDataModule(hparams)
    
    # Train
    trainer.fit(model, data_module)
