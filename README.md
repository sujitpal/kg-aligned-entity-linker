# Knowledge Graph Aligned Entity Linker using Sentence Transformers

## Abstract

This repository contains code to train and evaluate various Named Entity Linker (NEL) models. The NEL models are all pre-trained BERT-based transformers trained using contrastive learning using Sentence Transformers. The data used to train these models come from entity synonyms in a Knowledge Graph. During training, these NEL models learn to generate encodings that pushes synonyms closer together and non-synonyms further apart in the resulting embedding space.

## Introduction

NEL Models are deployed as part of a Named Entity Recognition and Linking (NERL) pipeline. In such pipelines, there is a Named Entity Recognizer (NER) that recognizes spans of text in the input as entities versus non-entities, followed by an NEL model that maps the recognized spans to entities in the Knowledge Graph. Standalone NER models are either dictionary-based, which can match Knowledge Graph synonyms exactly or with a certain degree of lexical fuzziness (lowercased and lemmatized matching, for example), or model based and capable of matching a small number of entities (for example PERson, ORGanization, etc.). The architecture of a NERL pipeline can not only handle recognizing the larger number of entities typically found in a Knowledge Graph, it can also handle semantic matching of discovered spans against entity synonyms in the Knowledge Graph.

NERL models are generally slower and less efficient compared to single stage dictionary-based NER models, but their ability to match spans of input text to Knowledge Graph entities in a semantic manner can make them useful in literature mining / entity discovery use cases.

## Methods

This work is inspired by [Self-Alignment Pretraining for Biomedical Entity Representations](https://arxiv.org/abs/2010.11784) (Liu et al, 2021) ([github](https://github.com/cambridgeltl/sapbert/tree/main)) which trains [PubmedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) using synonyms of entities from the [Unified Medical Language System](https://www.nlm.nih.gov/research/umls/index.html) (UMLS). Our model training also uses synonyms for UMLS entities but uses the contrastive training regime established by the [Sentence Transformers](https://www.sbert.net/) framework.

We fine-tune two pre-trained BERT models, [bert-base-uncased](https://huggingface.co/bert-base-uncased) and [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) (previously PubmedBERT) with entity synonyms from UMLS, using [MultipleNegativesRanking](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) (MNR) and [TripletLoss](https://www.sbert.net/docs/package_reference/losses.html#tripletloss) with Hard Negative Mining.

### Data Preparation

### Hard Negative Mining

### Evaluation

## Results

| MODEL_NAME        | POS_MEAN | POS_SD | NEG_MEAN | NEG_SD | DIFF |
| ----------------- | -------- | ------ | -------- | ------ | ---- |
| bert-base-uncased | 0.800    | 0.146  | 0.264    | 0.277 | 0.536 |
| **kgnel-bert-mnr<sup>[2]</sup>**    | **0.775**    | **0.182**  | **0.048**    | **0.071** | **0.728** |
| kognel-bert-trp   | 0.923    | 0.104  | 0.318    | 0.334 | 0.605 |
| kgnel-bert-trp-3  | 0.937    | 0.083  | 0.333    | 0.346 | 0.604 |
| kgnel-bert-trp-5  | 0.929    | 0.094  | 0.338    | 0.351 | 0.591 |
| **kgnel-bmbert-mnr<sup>[1]</sup>**  | **0.797**    | **0.185**  | **0.043**    | **0.065** | **0.754** |
| kgnel-bmbert-trp  | 0.930    | 0.093  | 0.322    | 0.334 | 0.609 |

### Visualizations

| MODEL NAME        | SCORE DISTRIBUTIONS                          | SCORE HEATMAP                                | DIFF  |
| ----------------- | -------------------------------------------- | -------------------------------------------- | ----- |
| bert-base-uncased | <img src="figs/dist-bert-base-uncased.png"/> | <img src="figs/heat-bert-base-uncased.png"/> | 0.536 |
| **kgnel-bert-mnr<sup>[2]</sup>**    | <img src="figs/dist-kgnel-bert-mnr.png "/>   | <img src="figs/heat-kgnel-bert-mnr.png"/>    | 0.728 |
| kgnel-bert-trp    | <img src="figs/dist-kgnel-bert-trp.png" />   | <img src="figs/heat-kgnel-bert-trp.png"/>    | 0.605 |
| kgnel-bert-trp-3  | <img src="figs/dist-kgnel-bert-trp-3.png"/>  | <img src="figs/heat-kgnel-bert-trp-3.png"/>  | 0.604 |
| kgnel-bert-trp-5  | <img src="figs/dist-kgnel-bert-trp-5.png"/>  | <img src="figs/heat-kgnel-bert-trp-5.png "/> | 0.591 |
| **kgnel-bmbert-mnr<sup>[1]</sup>**  | <img src="figs/dist-kgnel-bmbert-mnr.png"/>  | <img src="figs/heat-kgnel-bmbert-mnr.png"/>  | 0.754 |
| kgnel-bmbert-trp  | <img src="figs/dist-kgnel-bmbert-trp.png"/>  | <img src="figs/heat-kgnel-bmbert-trp.png"/>  | 0.609 |
 

## Running

01-explore-preprocess-data.ipynb

piython 01-split-trainset.py

python 02-vectorize-sampled-syns.py

python3 03-hard-neg-mining.py --num_neighbors 5
s
python 04-train-nel-st.py --input bert-base-uncased --loss trp --output ../data/kgnel-bert-trp-5 --triple-nn 5

python 05-eval-nel-st.py --model bert-base-uncased

p ython 060vil-nel-st.py --model bert-base-uncased

