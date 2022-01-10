"""
Script for computing the MRR scores over the test dataset for a pre-trained CodeQuery model
"""
from code_query.utils.testing import test_eval_setup

if __name__ == "__main__":
    trainer, model, data_module, _ = test_eval_setup()
    trainer.test(model, data_module)
