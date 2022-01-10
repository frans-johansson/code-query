"""
Utilities for testing and evaluation
"""
from argparse import ArgumentParser, Namespace
from typing import Tuple, Dict, List
from pathlib import Path

import torch
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from code_query.config import WANDB
from code_query.utils.logging import get_run_name
from code_query.utils.helpers import get_ckpt_dir, get_nl_dir
from code_query.model import CodeQuery, CodeQuerySiamese
from code_query.data import CSNetDataModule
from code_query.model.encoder import Encoder


def test_eval_setup() -> Tuple[Trainer, CodeQuery, CSNetDataModule, Namespace]:
    """
    Handles common setup for testing and evaluation scripts. Returning a tuple
    of the trainer, the model, the data module and the command line arguments namespace.
    """
    parser = ArgumentParser("CodeQuery indexing script")
    parser.add_argument("--wandb", action="store_true", help="Run the test against a W&B experiment")
    parser.add_argument("--run_id", type=str,
        help="Generated run ID for local checkpoints (e.g. 220106_1200) or a W&B run ID if used with the --wandb flag")
    parser.add_argument("--model_file", type=str,
        help="Name of a local .ckpt file (e.g. epoch=X-step=X.ckpt) or a W&B model version (e.g. v5 or best) if used with the --wandb flag")
    # parser.add_argument("--code_lang", type=str)
    # parser.add_argument("--query_langs", type=str, nargs="+", default=None)
    # parser.add_argument("--encoder_type", type=Encoder.Types)
    parser = CodeQuery.add_argparse_args(parser)
    parser = CSNetDataModule.add_argparse_args(parser)
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
    hparams = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]
    
    # Set up data module from checkpoint hyperparameters
    data_module = CSNetDataModule(hparams)

    # Load model checkpoint
    if "model_type" in hparams:
        model = CodeQuery.get_type(hparams["model_type"]).load_from_checkpoint(ckpt_path)
    else:
        # For models trained before the distinction between Siamese and Dual model types
        model = CodeQuerySiamese.load_from_checkpoint(ckpt_path)

    trainer = Trainer.from_argparse_args(args, enable_checkpointing=False, logger=logger)

    return (trainer, model, data_module, args)


def ndcg(predictions: Dict[str, List[str]], relevance_scores: Dict[str, Dict[str, float]],
         ignore_rank_of_non_annotated_urls: bool=True) -> float:
    """
    Computes the Normalized Discounted Cumulative Gain for a set of predictions and a set of relevance scores
    given as dictionaries. The predictions are expected to adhere to thethe following general schema
    `{"query": [urls, ...]}`, and the relevance scores as `{"query": {"url": relevance_score}}`, where
    
    * query: the textual representation of the query
    * url: the unique GitHub URL to the returned results

    The order of the URLs in the predictions imply the ranking of the results in the search task. Where URLs
    appearing earlier are assumed to have been given higher relevence.

    `ignore_rank_of_non_annotated_urls` controls whether to compute the "within" or "all" variant of the score.

    Code adapted from: https://github.com/github/CodeSearchNet/blob/master/src/relevanceeval.py
    under the MIT license.
    """
    num_results = 0
    ndcg_sum = 0

    for query, query_relevance_annotations in relevance_scores.items():
        current_rank = 1
        query_dcg = 0
        for url in predictions[query]:
            if url in query_relevance_annotations:
                query_dcg += (2**query_relevance_annotations[url] - 1) / np.log2(current_rank + 1)
                current_rank += 1
            elif not ignore_rank_of_non_annotated_urls:
                current_rank += 1

        query_idcg = 0
        for i, ideal_relevance in enumerate(sorted(query_relevance_annotations.values(), reverse=True), start=1):
            query_idcg += (2 ** ideal_relevance - 1) / np.log2(i + 1)
        if query_idcg == 0:
            # We have no positive annotations for the given query, so we should probably not penalize anyone about this.
            continue
        num_results += 1
        ndcg_sum += query_dcg / query_idcg
    return ndcg_sum / num_results
