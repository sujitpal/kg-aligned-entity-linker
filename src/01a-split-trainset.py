import numpy as np
import os

DATA_DIR = "../data"
INPUT_FILE = os.path.join(DATA_DIR, "positive_pairs.tsv")

TRAIN_OUTPUT = os.path.join(DATA_DIR, "positive_pairs_train.tsv")
TEST_OUTPUT = os.path.join(DATA_DIR, "positive_pairs_test.tsv")
VIZ_OUTPUT = os.path.join(DATA_DIR, "positive_pairs_viz.tsv")

NUM_TEST = 100
NUM_VIZ = 10

num_pos_pairs = 0
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("CUI"):
            continue
        num_pos_pairs += 1
print("num_input_pairs:", num_pos_pairs)

test_idxs = np.random.randint(0, num_pos_pairs, NUM_TEST)
test_idxs = set(test_idxs.tolist())
num_train, num_test = 0, 0
with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
        open(TRAIN_OUTPUT, "w", encoding="utf-8") as ftrain, \
        open(TEST_OUTPUT, "w", encoding="utf-8") as ftest:
    for i, line in enumerate(fin):
        if line.startswith("CUI"):
            ftrain.write(line)
            ftest.write(line)
            continue
        if i in test_idxs:
            ftest.write(line)
            num_test += 1
        else:
            ftrain.write(line)
            num_train += 1
print("num_train:", num_train, "num_test:", num_test)

viz_idxs = np.random.randint(0, NUM_TEST, NUM_VIZ)
viz_idxs = set(viz_idxs.tolist())
num_viz = 0
with open(TEST_OUTPUT, "r", encoding="utf-8") as fin, \
        open(VIZ_OUTPUT, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if line.startswith("CUI"):
            fout.write(line)
            continue
        if i in viz_idxs:
            fout.write(line)
            num_viz += 1
print("num_viz:", num_viz)
            