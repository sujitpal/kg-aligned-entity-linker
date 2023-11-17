import numpy as np
import os
import torch

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from transformers import AutoTokenizer, AutoModel


DATA_DIR = "../data"
INPUT_FILE = os.path.join(DATA_DIR, "positive_pairs_train.tsv")
BATCH_SIZE = 32

MODEL_NAME = "bert-base-uncased"


def encode_batch(syns, tokenizer, model):
    inputs = tokenizer(syns, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    encoding = outputs.pooler_output.cpu().numpy()
    encoding = encoding / np.linalg.norm(encoding, axis=1, keepdims=True)
    return encoding


def process_batch(batch, client, model_name, tokenizer, model,
                  already_indexed):
    cui_syns = [(id, cui, str_a) for id, cui, str_a, _ in batch]
    cui_syns.extend([(id+1, cui, str_p) for id, cui, _, str_b in batch])
    cui_syns = [x for x in cui_syns if x[2] not in already_indexed]
    vectors = encode_batch([syn for _, _, syn in cui_syns],
                           tokenizer, model)
    already_indexed.update([syn for _, _, syn in cui_syns])
    assert len(cui_syns) == vectors.shape[0]
    points = []
    for i in range(len(cui_syns)):
        id, cui, syn = cui_syns[i]
        vector = vectors[i]
        points.append(PointStruct(
            id=id, vector=vector.tolist(), payload={"cui": cui, "syn": syn}
        ))
    client.upsert(
        collection_name=model_name,
        wait=True,
        points=points)


if __name__ == "__main__":

    client = QdrantClient("localhost", port=6333)
    client.create_collection(
        collection_name=MODEL_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.DOT)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    num_pairs = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("CUI"):
                continue
            num_pairs += 1
    num_syns = num_pairs * 2

    already_indexed = set()
    batch = []
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            cui, str_a, str_p = line.strip().split("\t")
            if cui == "CUI":
                continue
            if i % 100 == 0:
                print("Processing {:d}/{:d} synonym pairs ({:.3f}%)".format(
                    i, num_pairs, i / num_pairs * 100))
            batch.append((i * 2 + 1, cui, str_a, str_p))
            if len(batch) == BATCH_SIZE:
                process_batch(batch, client, MODEL_NAME,
                              tokenizer, model, already_indexed)
                batch = []

    if len(batch) > 0:
        process_batch(batch, client, MODEL_NAME,
                      tokenizer, model, already_indexed)

    print("Processing {:d}/{:d} synonym pairs ({:.3f}%), COMPLETE".format(
        i, num_pairs, i / num_pairs * 100))

