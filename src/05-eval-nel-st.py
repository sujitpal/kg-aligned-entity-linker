import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from scipy.stats import norm
from sentence_transformers import SentenceTransformer, InputExample
from torch.nn.functional import cosine_similarity

DATA_DIR = "../data"
FIGURES_DIR = "../figs"
TEST_PAIRS_FILE = os.path.join(DATA_DIR, "positive_pairs_test.tsv")


def generate_points_for_normal_curve(mean, sd):
    x = np.linspace(mean - 3 * sd, mean + 3 * sd, 100)
    y = norm.pdf(x, mean, sd)
    return x, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or path")
    args = parser.parse_args()

    test_examples = []
    with open(TEST_PAIRS_FILE, "r", encoding="utf-8") as f:
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

    # # np.save("sim_matrix.npy", sim_matrix)
    # sim_matrix = np.load("sim_matrix_1.npy")

    # compute overlap between diagonal and off-diagonal elements
    diag_elements = np.diag(sim_matrix)
    mean_d = np.mean(diag_elements)
    std_d = np.std(diag_elements)

    off_diag_elements = np.tril(sim_matrix, k=-1)
    mean_od = np.mean(off_diag_elements)
    std_od = np.std(off_diag_elements)

    model_name = os.path.basename(args.model)
    print("| MODEL_NAME | POS_MEAN | POS_SD | NEG_MEAN | NEG_SD | DIFF |")
    print("| {:s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        model_name,
        mean_d, std_d, mean_od, std_od,
        mean_d - mean_od))

    xs_d, ys_d = generate_points_for_normal_curve(mean_d, std_d)
    xs_od, ys_od = generate_points_for_normal_curve(mean_od, std_od)

    plt.plot(xs_d, ys_d, label="positive")
    plt.plot(xs_od, ys_od, label="negative")
    plt.title(model_name)
    plt.xlabel("cosine similarity")
    plt.legend(loc="best")
    # _ = plt.show()
    dist_file = os.path.join(FIGURES_DIR, "dist-{:s}.png".format(model_name))
    plt.savefig(dist_file)
