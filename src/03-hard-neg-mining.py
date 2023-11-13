import numpy as np
import os
import torch

from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionStatus
from transformers import AutoTokenizer, AutoModel


DATA_DIR = "../data"
INPUT_FILE = os.path.join(DATA_DIR, "positive_pairs.tsv")
OUTPUT_FILE = os.path.join(DATA_DIR, "syn-triples.tsv")

MODEL_NAME = "bert-base-uncased"


def encode_string(s, tokenizer, model):
    inputs = tokenizer([s], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    encoding = outputs.pooler_output.cpu().numpy()[0]
    encoding = encoding / np.linalg.norm(encoding)
    return encoding


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

client = QdrantClient("localhost", port=6333)
coll_info = client.get_collection(collection_name=MODEL_NAME)
assert coll_info.status == CollectionStatus.GREEN

num_pairs = 0
with open(INPUT_FILE, "r", encoding="utf-8") as fin:
    for line in fin:
        if line.startswith("CUI"):
            continue
        num_pairs += 1

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    fout.write("\t".join(["CUI", "STR_A", "STR_P", "STR_N"]) + "\n")
    for lid, line in enumerate(fin):
        if line.startswith("CUI"):
            continue
        if lid % 100 == 0:
            print("Processing {:d}/{:d} rows ({:.3f}) %".format(
                lid, num_pairs, lid / num_pairs * 100))
        cui, str_a, str_p = line.strip().split("\t")
        qvec = encode_string(str_a, tokenizer, model)
        neighbors = client.search(
            collection_name=MODEL_NAME,
            query_vector=qvec.tolist(),
            limit=10
        )
        str_n = None
        for neighbor in neighbors:
            if neighbor.payload["cui"] == cui:
                continue
            str_n = neighbor.payload["syn"]
            break
        fout.write("\t".join([cui, str_a, str_p, str_n]) + "\n")

print("Processing {:d}/{:d} rows ({:.3f}) %, COMPLETE".format(
    lid, num_pairs, lid / num_pairs * 100))
