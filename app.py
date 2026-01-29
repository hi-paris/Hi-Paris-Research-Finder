import sys
from pathlib import Path
import ast
import pickle

import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# ROOT DIR for backend imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.exact_matching import exact_match

# Page config
st.set_page_config(
    page_title="Hi! Paris Research Profiles",
    page_icon=r"logo/icon_hi_search.png",
    layout="wide",
)

# Load data
file_path = "data/professor_profile_28_01.xlsx"
df = pd.read_excel(file_path)

df["Research domains"] = df["Research domains"].apply(ast.literal_eval)

affiliations = df["Affiliation"].dropna().unique().tolist()
df = df[df["Research axis"] != "not found"]
df = df.reset_index(drop=True)
# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-base-v2")

model = load_model()

# Load or build FAISS index
def build_prof_text(row):
    domains = row["Research domains"]
    domains_text = ", ".join(domains) if isinstance(domains, list) else str(domains)
    return (
        f"passage: Research axis: {row.get('Research axis', '')}. "
        f"Research domains: {domains_text}. "
        f"Summary: {row.get('Summary', '')}"
    )

try:
    with open("professor_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    required_keys = {"embeddings", "df", "ids"}
    if not required_keys.issubset(data):
        raise ValueError("Outdated pickle")

    embeddings = data["embeddings"]
    df_index = data["df"]
    ids = data["ids"]

    index = faiss.read_index("professor_index.faiss")

except Exception:
    st.warning("Rebuilding FAISS index…")

    df_index = df.copy()
    ids = df_index["ID"].tolist()

    texts = [build_prof_text(row) for _, row in df_index.iterrows()]
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "professor_index.faiss")

    with open("professor_embeddings.pkl", "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "df": df_index,
                "ids": ids,
            },
            f,
        )

# Helpers
def label_match(score):
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"

# Semantic search
def semantic_search_fast(query, threshold=0.4, top_k=10, exact_ids=None):
    if exact_ids is None:
        exact_ids = set()

    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        score = float(score)
        prof_id = ids[idx]

        if score < threshold or prof_id in exact_ids:
            continue

        row = df_index[df_index["ID"] == prof_id].iloc[0]

        results.append(
            {
                "ID": prof_id,
                "Match Level": label_match(score),
                "Last name": row.get("Last name", ""),
                "First name": row.get("First name", ""),
                "Affiliation": row.get("Affiliation", ""),
                "Research axis": row.get("Research axis", ""),
                "Research domains": row.get("Research domains", ""),
                "Summary": row.get("Summary", ""),
                "FAISS Score": score,
            }
        )

    return pd.DataFrame(results)

# Streamlit UI
st.title("Hi! Paris Expert Finder")
st.markdown(
    "Enter a research domain below to find all relevant researcher profiles affiliated with Hi! Paris."
)

st.sidebar.image(r"logo/logo_hi_paris.png", width=300)
st.sidebar.header("Options")

show_exact = st.sidebar.checkbox("Show exact matches", value=True)
show_suggestions = st.sidebar.checkbox("Show suggestions", value=True)

selected_affiliations = st.sidebar.multiselect(
    "Filter by affiliation:", options=affiliations
)

query = st.text_input(
    "Enter a research domain:",
    placeholder="e.g. optimal transport, time series, computer vision",
)

# Main logic
if not query:
    if selected_affiliations:
        df_filtered = df[df["Affiliation"].isin(selected_affiliations)]
        df_filtered.index += 1
        st.subheader("Researchers by affiliation")
        st.dataframe(
            df_filtered[
                [
                    "Last name",
                    "First name",
                    "Affiliation",
                    "Research axis",
                    "Research domains",
                    "Summary",
                ]
            ],
            width="stretch",
        )
    else:
        st.info("Please enter a research domain or select an affiliation.")
        df.index += 1
        st.dataframe(
            df[
                [
                    "Last name",
                    "First name",
                    "Affiliation",
                    "Research axis",
                    "Research domains",
                    "Summary",
                ]
            ],
            width="stretch",
        )
else:
    df_filtered = (
        df[df["Affiliation"].isin(selected_affiliations)]
        if selected_affiliations
        else df
    )

    exact_results = exact_match(query, df_filtered)
    exact_results.index += 1
    exact_ids = set(exact_results["ID"].tolist()) if not exact_results.empty else set()

    if show_exact:
        st.subheader("Exact matches")
        if exact_results.empty:
            st.info("No exact match found")
        else:
            st.dataframe(
                exact_results[
                    [
                        "Last name",
                        "First name",
                        "Affiliation",
                        "Research axis",
                        "Research domains",
                        "Summary",
                    ]
                ],
                width="stretch",
            )

    if show_suggestions:
        suggestions = semantic_search_fast(query, exact_ids=exact_ids)
        suggestions.index += 1
        st.subheader("Suggestions")
        if suggestions.empty:
            st.info("No suggestions found")
        else:
            st.dataframe(suggestions, width="stretch")
