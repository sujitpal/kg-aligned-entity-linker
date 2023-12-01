import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from sentence_transformers import SentenceTransformer, InputExample
from torch.nn.functional import cosine_similarity


DATA_DIR = "../data"
FIGURES_DIR = "../figs"

VIZ_PAIRS_FILE = os.path.join(DATA_DIR, "positive_pairs_viz.tsv")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or path")
    args = parser.parse_args()

    test_examples = []
    with open(VIZ_PAIRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CUI"):
                continue
            _, str_a, str_p = line.strip().split("\t")
            test_examples.append(InputExample(texts=[str_a, str_p], label=1.0))
    print("#-examples:", len(test_examples))

    # evaluation
    best_model = SentenceTransformer(args.model)
    lhs_vectors, rhs_vectors = [], []
    for example in test_examples:
        lhs_vectors.append(
            best_model.encode(example.texts[0], convert_to_tensor=True))
        rhs_vectors.append(
            best_model.encode(example.texts[1], convert_to_tensor=True))

    sim_matrix = np.zeros((len(lhs_vectors), len(rhs_vectors)))
    for i, lhs_vector in enumerate(lhs_vectors):
        for j, rhs_vector in enumerate(rhs_vectors):
            if i < j:
                continue
            sim_matrix[i, j] = cosine_similarity(lhs_vector, rhs_vector, dim=0)

    # np.save("sim_matrix.npy", sim_matrix)
    # sim_matrix = np.load("sim_matrix_1.npy")

    model_name = os.path.basename(args.model)
    heatmap_file = os.path.join(FIGURES_DIR, "heat-{:s}.png".format(model_name))
    # heatmap (sim_matrix is lower triangular)
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    data = np.ma.array(sim_matrix, mask=mask)
    plt.title(model_name)
    plt.xlabel("LHS synonynm")
    plt.ylabel("RHS synonynm")
    plt.imshow(data, cmap="Blues", interpolation="nearest")
    # _ = plt.show()
    plt.savefig(heatmap_file)
