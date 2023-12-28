import json
import numpy as np
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


DATA_DIR = "../data"
INPUT_SYNS_PATH = os.path.join(DATA_DIR, "syns_by_cui.jsonl")
MODEL_PATH = os.path.join(DATA_DIR, "kgnel-bmbert-mnr")
COLLECTION_NAME = "nel-kgnel-bmbert-mnr"
NUM_TOTAL = 250117  # wc -l syns_by_cui.jsonl

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.DOT)
)

num_inserted = 0
model = SentenceTransformer(MODEL_PATH)
with open(INPUT_SYNS_PATH, "r", encoding="utf-8") as fin:
    for id, line in enumerate(fin):
        if num_inserted % 100 == 0:
            print("{:d}/{:d} ({:.3f}%) done".format(
                num_inserted, NUM_TOTAL, num_inserted * 100 / NUM_TOTAL))
        # parse JSON line
        line_json = json.loads(line.strip())
        cui = line_json["CUI"]
        sty = line_json["STY"]
        syns = line_json["STR"]
        # encode all synonyms and L2-normalize (since using DOT)
        vecs = model.encode(syns)
        l2_norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / l2_norm
        # compute centroid for each CUI
        centroid = np.mean(vecs, axis=0).tolist()
        payload = {
            "cui": line_json["CUI"],
            "cfn": syns[0],
            "sty": line_json["STY"],
        }
        # convert into QDrant PointStructs and insert
        points = [PointStruct(id=id, vector=centroid, payload=payload)]
        op_info = client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        num_inserted += 1

print("{:d}/{:d} ({:.3f}%) done, COMPLETE".format(
    num_inserted, NUM_TOTAL, num_inserted * 100 / NUM_TOTAL))
