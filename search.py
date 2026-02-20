import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

from utils import build_prof_text, label_match


@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-base-v2")


def load_or_build_index(df, excel_hash):
    model = load_model()

    try:
        with open("professor_embeddings.pkl", "rb") as f:
            data = pickle.load(f)

        if data.get("excel_hash") != excel_hash:
            raise ValueError("Data changed, need to rebuild embeddings")

        embeddings = data["embeddings"]
        df_copy = data["df"]
        ids = data["ids"]
        index = faiss.read_index("professor_index.faiss")

    except Exception:
        df_copy = df.copy()
        ids = df_copy["ID"].tolist()
        texts = [build_prof_text(row) for _, row in df_copy.iterrows()]
        embeddings = model.encode(texts, normalize_embeddings=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, "professor_index.faiss")

        with open("professor_embeddings.pkl", "wb") as f:
            pickle.dump(
                {
                    "embeddings": embeddings,
                    "df": df_copy,
                    "ids": ids,
                    "excel_hash": excel_hash,
                },
                f,
            )

    return model, index, df_copy, ids


def exact_match(query, df):
    q = query.strip().lower()

    matched_df = df[
        df["Research domains"].apply(lambda ds: any(q in d.lower().strip() for d in ds))
    ]

    return matched_df.reset_index(drop=True)


def semantic_search_fast(
    query, model, index, df_copy, ids,
    threshold=0.0, top_k=10, exact_ids=None, allowed_axes=None
):
    if exact_ids is None:
        exact_ids = set()

    q_emb = model.encode(
        [f"query: professor working on {query}"], normalize_embeddings=True
    )

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        score = float(score)
        prof_id = ids[idx]

        if score < threshold or prof_id in exact_ids:
            continue

        row = df_copy[df_copy["ID"] == prof_id].iloc[0]

        if allowed_axes and row["Research axis"] not in allowed_axes:
            continue

        results.append(
            {
                "Match Level": label_match(score),
                "Last name": row.get("Last name", ""),
                "First name": row.get("First name", ""),
                "Affiliation": row.get("Affiliation", ""),
                "Research axis": row.get("Research axis", ""),
                "Research domains": row.get("Research domains", ""),
                "Summary": row.get("Summary", ""),
                "Personal webpage": row.get("Personal webpage", ""),
            }
        )

    return pd.DataFrame(results)