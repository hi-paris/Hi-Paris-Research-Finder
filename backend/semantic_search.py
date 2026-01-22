# backend/semantic_search.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the mdeol
model = SentenceTransformer("all-MiniLM-L6-v2")


def label_match(score):
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"


def semantic_search_filtered(query, df, exact_ids=set(), threshold=0.4, top_k=10):
    df_filtered = df[~df["ID"].isin(exact_ids)].reset_index(
        drop=True
    )  # Drop Exact matches
    if df_filtered.empty:
        return pd.DataFrame([])

    texts = [
        " ".join(domains) for domains in df_filtered["Research domains"]
    ]  # Encode all the reseach domains
    # texts = [" ".join(ds[:5]) for ds in df_filtered["research domains"]]  # Encode only the first 5 reseach domains

    embeddings = model.encode(texts, normalize_embeddings=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Encod the query
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            row = df_filtered.iloc[idx]
            results.append(
                {
                    "Match Level": label_match(float(score)),
                    "Last name": row.get("Last name", ""),
                    "First name": row.get("First name", ""),
                    "Affiliation": row.get("Affiliation", ""),
                    "Research axis": row.get("Research axis", ""),
                    "Research domains": row.get("Research domains", ""),
                    "Summary": row.get("Summary", ""),
                }
            )
    return pd.DataFrame(results)
