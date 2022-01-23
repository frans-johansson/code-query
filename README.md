# Code Query 🔎
**A project about retrieving relevant source code examples given a natural language query.**

---

[![TDDE16](https://img.shields.io/badge/LiU-TDDE16-blue)](https://www.ida.liu.se/~TDDE16/)
[![Python +3.8](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org)
[![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.10-orange)](https://pytorch.org/)
[![MIT](https://img.shields.io/badge/license-MIT-green)](https://pytorch.org/)

This project examines approaches to semantic code search using embeddings from deep neural networks such as BERT. The main architecture was insipired by the baselines proposed by [Husain et al.](https://arxiv.org/pdf/1909.09436.pdf)., which also provide the [CodeSearchNet](https://github.com/github/CodeSearchNet) corpus used to train the models in this project.

The project was carried out as the graded, individual work for the course TDDE16, Text Mining at Linköping University

## Quickstart 🐎

First, make sure to create a virtual environment `python -m venv venv`
Then activate it

- For Windows: `.\venv\Scripts\activate`
- For Linux/Mac OS: `source ./venv/bin/activate`

Install all required dependencies `pip install -r requirements.txt`

> ⚠️ In addition to the requirements installed from the file, you must also make sure to install the local `code_query` module by invoking `pip install -e ./src` from the root directory

Get started training a simple Neural Bag of Words (NBOW) model by running `python ./scripts/train.py --model_type siamese --encoder_type nbow --code_lang python --max_epochs 10`

## Scripts ✒️
The `./scripts` directory holds Python scripts for training, hyperparameter tuning (intended to be used with [Weights and Biases sweeps](https://docs.wandb.ai/guides/sweeps)), testing and evaluating the models.

### Training
Running `python ./scripts/train.py` will train a model using PyTorch and PyTorch Lightning. You will need to supply the following arguments:

- `--model_type`: Either `siamese` for shared encoder weights or `dual` for separate
- `--encoder_type`: Specifies the encoder type. Supports `nbow`, `bert`, `codebert`, `distilbert`, `roberta`. Additional BERT-likes can be configured by updating `config/models.yml` and the `code_query.model.encoder.Encoder.Types` enum.
- `--code_lang`: One of the supported CodeSearchNet programming languages

Additionally, the following arguments might be interesting to specify:

- `--query_langs`: To perform pre-process filtering on a list of natural languages using [fastText](https://fasttext.cc/).
- `--embedding_dim`: Primarily used for the `nbow` encoder type, and specifies the dimensions of the embeddings.
- `--encoding_dim`: Sets the dimension of the densely projected embeddings to the final encodings.

Additional arguments are provided by the PyTorch Lightning `Trainer` API. E.g. to train on a single GPU simply add `--gpus 1`.

### Testing and evaluating
To evaluate a trained model on the test split of the dataset, run `python ./scripts/test.py`. This will compute the [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) score of the model on the test data.

To evaluate using [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) against expert relevance annotations, run `python ./scripts/eval.py`.

Both of these scripts require at least the following arguments:

- `--run_id`: The generated ID of the run you want to run on. If Weights and Biases is not used, this will correspond to a local directory name under `runs/ckpts/[code lang]/[query langs|all]/ID`, such as `220106_1200`
- `--model_file`: A file name, or model version if using Weights and Baises. Locally this corresponds to a `.ckpt` file formatted something like `epoch=X-step=X.ckpt`

> ⚠️ You may run into issues if your set up regarding GPUs etc. does not match the training phase. If this occurs, you might have too supply additional arguments to match the set ups. Note that these scripts accept the same arguments as the training script in addition to the ones specified above.

## Configuration ⚙️
For project level configuration, and other stuff not handled by the script arguments, please have a look at the `.yml` files in the `./config` directory.

## References 📚
A few relevant articles on the subject of semantic code search

- Husain, Hamel, et al. "Codesearchnet challenge: Evaluating the state of semantic code search." arXiv preprint arXiv:1909.09436 (2019).
- Feng, Zhangyin, et al. "Codebert: A pre-trained model for programming and natural languages." arXiv preprint arXiv:2002.08155 (2020).
- Schütze, Hinrich, Christopher D. Manning, and Prabhakar Raghavan. Introduction to information retrieval. Vol. 39. Cambridge: Cambridge University Press, 2008.
