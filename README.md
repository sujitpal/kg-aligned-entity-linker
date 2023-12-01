# kg-aligned-entity-linker
Knowledge Graph Aligned Entity Linker using BERT and Sentence Transformers

01-explore-preprocess-data.ipynb

python 01-split-trainset.py

python 02-vectorize-sampled-syns.py

python3 03-hard-neg-mining.py --num_neighbors 5

python 04-train-nel-st.py --input bert-base-uncased --loss trp --output ../data/kgnel-bert-trp-5 --triple-nn 5

python 05-eval-nel-st.py --model bert-base-uncased

python 060vil-nel-st.py --model bert-base-uncased

## Evaluation

| MODEL_NAME | POS_MEAN | POS_SD | NEG_MEAN | NEG_SD | DIFF |
| ---------- | -------- | ------ | -------- | ------ | ---- |
| bert-base-uncased | 0.800 | 0.146 | 0.264 | 0.277 | 0.536 |
| kgnel-bert-mnr | 0.775 | 0.182 | 0.048 | 0.071 | 0.728 |
| kgnel-bert-trp | 0.923 | 0.104 | 0.318 | 0.334 | 0.605 |
| kgnel-bert-trp-3 | 0.937 | 0.083 | 0.333 | 0.346 | 0.604 |
| kgnel-bert-trp-5 | 0.929 | 0.094 | 0.338 | 0.351 | 0.591 |
| kgnel-bmbert-mnr | 0.797 | 0.185 | 0.043 | 0.065 | 0.754 |
| kgnel-bmbert-trp | 0.930 | 0.093 | 0.322 | 0.334 | 0.609 |



