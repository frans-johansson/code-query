"""
Indexes the entire evaluation dataset for relevance evaluation (NDCG) and general purpose querying
"""
import wandb
from torch.utils.data.dataloader import DataLoader

from code_query.data import CSNetDataset, CSNetTokenizerManager, read_relevance_annotations
from code_query.utils.helpers import get_ann_dir
from code_query.utils.testing import ndcg, test_eval_setup
from code_query.config import TRAINING


if __name__ == "__main__":
    trainer, model, data_module, args = test_eval_setup()

    # Set up indexing
    eval_dataset = CSNetDataset(
        args.encoder_type.value,
        args.code_lang,
        args.query_langs,
        training=False
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    model.setup_index(
        eval_dataloader,
        # TODO: Move to config
        n_trees=200,
        ann_dir=get_ann_dir(
            encoder_type=args.encoder_type,
            code_lang=args.code_lang,
            query_langs=args.query_langs,
            run_id=args.run_id
        )
    )

    # Set up tokenizing and lookups for test queries
    tokenizer = CSNetTokenizerManager(
        model_name=args.encoder_type.value,
        code_lang=args.code_lang,
        tiny=False,
        query_langs=args.query_langs
    ).get_tokenizer(TRAINING.SEQUENCE_LENGTHS.QUERY)
    model.setup_predictor(
        eval_lookup=eval_dataset._source_data,
        tokenizer=tokenizer
    )

    # Get predictions
    res = trainer.predict(model, data_module)
    preds = {query: urls for batch in res for query, urls in batch.items()}
    
    # Read in relevance annotations and compute the NDCG score
    rel_annot = read_relevance_annotations(args.code_lang)
    within_score = ndcg(preds, rel_annot, ignore_rank_of_non_annotated_urls=False)
    all_score = ndcg(preds, rel_annot, ignore_rank_of_non_annotated_urls=True)
    
    # Log to wandb via the model
    if args.wandb:
        wandb.log({"eval/ndcg_within": within_score})
        wandb.log({"eval/ndcg_all": all_score})
    else:
        print(f"{within_score=}")
        print(f"{all_score=}")
