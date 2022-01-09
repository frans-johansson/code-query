"""
Script for computing the MRR scores over the test dataset for a pre-trained CodeQuery model
"""
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from code_query.data import CSNetDataModule
from code_query.model import CodeQuery, CodeQuerySiamese
from code_query.model.encoder import Encoder
from code_query.utils.helpers import get_ckpt_dir, get_nl_dir
from code_query.utils.logging import get_run_name
from code_query.config import WANDB

if __name__ == "__main__":
    parser = ArgumentParser("CodeQuery test script")
    parser.add_argument("--wandb", action="store_true", help="Run the test against a W&B experiment")
    parser.add_argument("--run_id", type=str,
        help="Generated run ID for local checkpoints (e.g. 220106_1200) or a W&B run ID if used with the --wandb flag")
    parser.add_argument("--model_file", type=str,
        help="Name of a local .ckpt file (e.g. epoch=X-step=X.ckpt) or a W&B model version (e.g. v5 or best) if used with the --wandb flag")
    parser.add_argument("--code_lang", type=str)
    parser.add_argument("--query_langs", type=str, nargs="+", default=None)
    parser.add_argument("--encoder_type", type=Encoder.Types)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Weights and Biases logger
    logger = None
    if args.wandb:
        wandb_dir = Path(WANDB.DIR)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger = WandbLogger(
            name=get_run_name(args.encoder_type),
            project=WANDB.PROJECT_NAME,
            save_dir=wandb_dir,
            log_model=False,
            tags=[args.encoder_type.value, args.code_lang, str(get_nl_dir(args.query_langs)), "Test"]
        )
        # Set up model checkpoint from run artifact
        artifact = logger.experiment.use_artifact(f"model-{args.run_id}:{args.model_file}", type="model")
        ckpt_path = Path(artifact.download()) / "model.ckpt"
    else:
        # Define the location of the checkpoint file locally
        ckpt_path = get_ckpt_dir(
            code_lang=args.code_lang,
            encoder_type=args.encoder_type,
            query_langs=args.query_langs,
            run_id=args.run_id
        ) / args.model_file
    
    # Loads hyperparameters from the checkpoint file
    hparams = torch.load(ckpt_path)["hyper_parameters"]
    
    # Set up data module from checkpoint hyperparameters
    data_module = CSNetDataModule(hparams)

    # Load model checkpoint
    if "model_type" in hparams:
        model = CodeQuery.get_type(hparams["model_type"]).load_from_checkpoint(ckpt_path)
    else:
        # For models trained before the distinction between Siamese and Dual model types
        model = CodeQuerySiamese.load_from_checkpoint(ckpt_path)

    trainer = Trainer.from_argparse_args(args, enable_checkpointing=False, logger=logger)
    trainer.test(model, data_module)
