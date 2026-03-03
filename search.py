import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import Any
from collections.abc import Sequence
from utils import build_prof_text, label_match


@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-base-v2")


import pickle
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd
import faiss


def load_or_build_index(
    df: pd.DataFrame,
    excel_hash: str,
) -> tuple[Any, faiss.Index, pd.DataFrame, list]:
    """
    Load an existing FAISS index + embeddings from disk if they match the current dataset hash,
    otherwise rebuild embeddings + index and persist them.
    """
    model = load_model()

    pkl_path = "professor_embeddings.pkl"
    index_path = "professor_index.faiss"

    # Try loading existing artifacts
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError("Invalid pickle format: expected a dict.")

        if data.get("excel_hash") != excel_hash:
            raise ValueError("Data changed: need to rebuild embeddings/index.")

        embeddings = data["embeddings"]
        df_copy = data["df"]
        ids = data["ids"]

        # Basic validation
        if not isinstance(df_copy, pd.DataFrame):
            raise ValueError("Invalid pickle content: 'df' is not a DataFrame.")

        if not isinstance(ids, list):
            raise ValueError("Invalid pickle content: 'ids' is not a list.")

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        # FAISS expects float32
        embeddings = embeddings.astype(np.float32, copy=False)

        index = faiss.read_index(index_path)

        if index.ntotal != embeddings.shape[0]:
            raise ValueError("Index size does not match embeddings; rebuild required.")

    except (FileNotFoundError, EOFError, pickle.UnpicklingError, KeyError, ValueError):
        # Rebuild
        df_copy = df.copy()

        if "ID" not in df_copy.columns:
            raise KeyError("Input DataFrame must contain an 'ID' column.")

        ids = df_copy["ID"].tolist()
        texts = [build_prof_text(row) for _, row in df_copy.iterrows()]

        embeddings = model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, index_path)

        with open(pkl_path, "wb") as f:
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


import pandas as pd


def exact_match(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows where the query exactly matches one of
    the research domains.
    """
    q = query.strip().lower()

    matched_df = df[
        df["Research domains"].apply(
            lambda ds: isinstance(ds, list) and any(q in d.lower().strip() for d in ds)
        )
    ]

    return matched_df.reset_index(drop=True)





def semantic_search_fast(
        query: str,
        model: Any,
        index: faiss.Index,
        df_copy: pd.DataFrame,
        ids: Sequence[Any],
        threshold: float = 0.0,
        top_k: int = 10,
        exact_ids: set[Any] | None = None,
        allowed_affiliations: set[str] | None = None,
        allowed_axes: set[str] | None = None,
        ) -> pd.DataFrame:
    """
    Run a semantic search over a FAISS index and return matching professor records.
    """
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
        if allowed_affiliations and row["Affiliation"] not in allowed_affiliations:
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