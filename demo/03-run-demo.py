import numpy as np
import os
import pandas as pd
import spacy
import streamlit as st

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from spacy import displacy


DATA_DIR = "../data"
NER_MODEL = "en_core_sci_sm"
NEL_MODEL = "kgnel-bmbert-mnr"
COLLECTION_NAME = "nel-kgnel-bmbert-mnr"
TOP_K = 3
SCORE_THRESHOLD = 0.7


@st.cache_resource
def load_ner_model():
    ner_model = spacy.load(NER_MODEL)
    return ner_model


@st.cache_resource
def load_nel_model():
    nel_model = SentenceTransformer(os.path.join(DATA_DIR, NEL_MODEL))
    return nel_model


@st.cache_resource
def connect_to_qdrant():
    client = QdrantClient("localhost", port=6333)
    return client


def extract_entities_and_rendered_text(text, model):
    doc = model(text)
    token_char_offsets = {}
    for tok in doc:
        token_char_offsets[tok.i] = (tok.idx, tok.idx + len(tok.text))
    rows = []
    for ent in doc.ents:
        rows.append({
            "token_text": ent.text,
            "tok_start": ent.start,
            "tok_end": ent.end,
            "char_start": token_char_offsets[ent.start][0],
            "char_end": token_char_offsets[ent.end][1],
            "ent_type": ent.label_,
        })
    ents_df = pd.DataFrame(rows)
    html = displacy.render(doc, style="ent", page=True)
    return ents_df, html, doc


def link_entities(ents_df, model, qdrant_client):
    phrases = ents_df.token_text.values
    # handle duplicates to minimize calls to NEL
    unique_phrases = list(set(phrases))
    vecs = model.encode(unique_phrases)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    linked_entities = {}
    for phrase, vec in zip(unique_phrases, vecs):
        result = qdrant_client.search(collection_name=COLLECTION_NAME,
                                      query_vector=vec,
                                      with_payload=True,
                                      limit=TOP_K)
        entities = []
        for scored_point in result:
            score = scored_point.score
            payload = scored_point.payload
            entities.append((payload, score))
        linked_entities[phrase] = entities
    ents_df = ents_df.assign(pred_cui=["N/A"] * len(ents_df))
    ents_df = ents_df.assign(pred_cfn=["N/A"] * len(ents_df))
    ents_df = ents_df.assign(score=[0.0] * len(ents_df))
    for phrase, entities in linked_entities.items():
        pred_payload, score = entities[0]
        pred_cui = pred_payload["cui"]
        pred_cfn = pred_payload["cfn"]
        pred_ent_type = pred_payload["sty"]
        ents_df.loc[ents_df["token_text"] == phrase, "ent_type"] = \
            "DISEASE" if pred_ent_type == "Disease or Syndrome" else "DRUG"
        ents_df.loc[ents_df["token_text"] == phrase, "score"] = \
            score
        ents_df.loc[ents_df["token_text"] == phrase, "pred_cui"] = \
            pred_cui
        ents_df.loc[ents_df["token_text"] == phrase, "pred_cfn"] = \
            pred_cfn
    return ents_df


def render_updated_links(doc, ents_df):
    ent_labels = ents_df.ent_type.values
    scores = ents_df.score.values
    ents = []
    for ent, ent_label, score in zip(doc.ents, ent_labels, scores):
        if score < SCORE_THRESHOLD:
            continue
        ent.label_ = ent_label
        ents.append(ent)
    doc.ents = ents
    options = {
        "ents": ["DRUG", "DISEASE"],
        "colors": {
            "DRUG": "#FF5733",
            "DISEASE": "#3498DB"
        }
    }
    html = displacy.render(doc, style="ent", page=True, options=options)
    return html


if __name__ == "__main__":
    ner_model = load_ner_model()
    nel_model = load_nel_model()
    qdrant = connect_to_qdrant()

    st.title("Knowledge Graph Aligned Entity Linker Demo")
    input_text = st.text_area("Enter Text to Link", height=10)
    nerl_this = st.button("NERL this!")
    if nerl_this:
        ents_df, ner_html, doc = extract_entities_and_rendered_text(
            input_text, ner_model)
        st.markdown("### Named Entity Recognition (NER) Output")
        st.dataframe(ents_df)
        st.markdown(ner_html, unsafe_allow_html=True)
        st.markdown("### Named Entity Linking (NEL) Output")
        ents_df = link_entities(ents_df, nel_model, qdrant)
        st.dataframe(ents_df)
        nel_html = render_updated_links(doc, ents_df)
        st.markdown(nel_html, unsafe_allow_html=True)
