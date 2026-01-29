import faiss
import pickle
import pandas as pd

# ---------- Load FAISS index and embeddings once ----------
def load_faiss_index(index_path="models/professor_index.faiss",
                     embeddings_path="models/professor_embeddings.pkl"):
    index = faiss.read_index(index_path)
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    df = data["df"]
    return index, embeddings, df


# ---------- Label matches ----------
def label_match(score: float) -> str:
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"


# ---------- Semantic search ----------
def semantic_search(model, query, index, df, threshold=0.4, top_k=10):
    """Return top matching professors based on FAISS similarity."""
    # Encode query
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)

    # FAISS search
    scores, indices = index.search(q_emb, top_k)

    # Collect results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            row = df.iloc[idx]
            results.append({
                "Match Level": label_match(float(score)),
                "Last name": row.get("Last name", ""),
                "First name": row.get("First name", ""),
                "Affiliation": row.get("Affiliation", ""),
                "Research axis": row.get("Research axis", ""),
                "Research domains": row.get("Research domains", ""),
                "Summary": row.get("Summary", ""),
                "FAISS Score": float(score)
            })
    return pd.DataFrame(results)
