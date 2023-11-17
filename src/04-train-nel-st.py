import argparse
import os
import torch

from sentence_transformers import (
    InputExample, SentenceTransformer, losses, models
)
from torch.utils.data import DataLoader


DATA_DIR = "../data"
POSITIVE_PAIRS_FILE = os.path.join(DATA_DIR, "positive_pairs_train.tsv")
SYN_TRIPLES_FILE = os.path.join(DATA_DIR, "syn_triples.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="model to fine-tune")
    parser.add_argument("--loss", choices=["mnr", "trp"], help="loss function")
    parser.add_argument("--output", type=str, help="model name or path")
    args = parser.parse_args()

    # model definition
    word_embedding_model = models.Transformer(args.input, max_seq_length=256)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=768,
        activation_function=torch.nn.ReLU())
    model = SentenceTransformer(modules=[
        word_embedding_model,
        pooling_model,
        dense_model])

    # loss function and data preparation
    train_examples = []
    if args.loss == "mnr":
        with open(os.path.join(DATA_DIR, "positive_pairs_train.tsv")) as f:
            for line in f:
                if line.startswith("CUI"):
                    continue
                _, str_a, str_p = line.strip().split("\t")
                train_examples.append(InputExample(texts=[str_a, str_p],
                                                   label=1.0))
        loss_fn = losses.MultipleNegativesRankingLoss(model)
    elif args.loss == "trp":
        with open(os.path.join(DATA_DIR, "syn_triples.tsv")) as f:
            for line in f:
                if line.startswith("CUI"):
                    continue
                _, str_a, str_p, str_n = line.strip().split("\t")
                train_examples.append(InputExample(texts=[str_a, str_p, str_n]))
        loss_fn = losses.TripletLoss(model)
    else:
        raise ValueError("Unknown loss function: {:s}".format(args.loss))

    # data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # training
    model.fit(train_objectives=[
            (train_dataloader, loss_fn)
        ],
        epochs=10,
        warmup_steps=100,
        output_path=args.output,
        save_best_model=True,
        show_progress_bar=True
    )
    print("model saved to {:s}".format(args.output))
