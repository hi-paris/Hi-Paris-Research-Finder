import ast
import hashlib
import pickle

import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# Hash function
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# Page config
st.set_page_config(
    page_title="Hi! Paris Research Profiles",
    page_icon=r"logo/icon_hi_search.png",
    layout="wide",
)

# Load data
file_path = r"data/Hi_Paris_affiliated_professors.xlsx"
df = pd.read_excel(file_path)
excel_hash = file_hash(file_path)


df["Research domains"] = df["Research domains"].apply(ast.literal_eval)
df = df[df["Research axis"] != "not found"].reset_index(drop=True)

# Options for filters
affiliations = df["Affiliation"].dropna().unique().tolist()
research_axis_options = df["Research axis"].dropna().unique().tolist()


# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-base-v2")


model = load_model()


# Load FAISS index
def build_prof_text(row):
    domains = row["Research domains"]
    domains_text = ", ".join(domains) if isinstance(domains, list) else str(domains)
    return (
        f"passage: Research domains: {domains_text}. "
        f"Summary: {row.get('Summary', '')}"
    )


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
    # st.warning("Rebuilding FAISS index…")
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


# Label match
def label_match(score):
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"


# Exact match function
def exact_match(query, df):
    q = query.strip().lower()

    matched_df = df[
        df["Research domains"].apply(lambda ds: any(q in d.lower().strip() for d in ds))
    ]

    return matched_df.reset_index(drop=True)


# Semantic search
def semantic_search_fast(
    query, threshold=0.0, top_k=10, exact_ids=None, allowed_axes=None
):
    if exact_ids is None:
        exact_ids = set()

    # q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
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


# Apply filters
def apply_filters(df, affiliations, research_axes):
    if affiliations:
        df = df[df["Affiliation"].isin(affiliations)]
    if research_axes:
        df = df[df["Research axis"].isin(research_axes)]
    return df


# Streamlit UI
st.title("Hi! Paris Expert Finder")
st.markdown(
    (
        "Enter a research domain below to find all relevant researcher profiles "
        "affiliated with Hi! Paris."
    )
)
st.sidebar.image(r"logo/logo_hi_paris.png", width=300)
st.sidebar.header("Options")

show_exact = st.sidebar.checkbox("Show exact matches", value=True)
show_suggestions = st.sidebar.checkbox("Show suggestions", value=True)

query = st.text_input(
    "Enter a research domain:",
    placeholder="e.g. optimal transport, time series, computer vision",
)

col1, col2 = st.columns(2)

with col1:
    selected_affiliations = st.multiselect(
        "Filter by affiliation:", options=affiliations
    )

with col2:
    selected_research_axis = st.multiselect(
        "Filter by Research axis:", options=research_axis_options
    )

# Main logic
df_filtered = apply_filters(df, selected_affiliations, selected_research_axis)

if not query:
    if df_filtered.empty:
        st.info("No researchers match the selected filters.")
    else:
        df_filtered.index += 1
        st.subheader("Researchers")
        st.dataframe(
            df_filtered[
                [
                    "Last name",
                    "First name",
                    "Affiliation",
                    "Research axis",
                    "Research domains",
                    "Summary",
                    "Personal webpage",
                ]
            ],
            width="stretch",
            column_config={
                "Personal webpage": st.column_config.LinkColumn(
                    display_text=r"https?://(.*)$"
                )
            },
        )
else:
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
                        "Personal webpage",
                    ]
                ],
                width="stretch",
                column_config={
                    "Personal webpage": st.column_config.LinkColumn(
                        display_text=r"https?://(.*)$"
                    )
                },
            )

    if show_suggestions:
        suggestions = semantic_search_fast(
            query,
            exact_ids=exact_ids,
            allowed_axes=selected_research_axis,
        )
        suggestions.index += 1
        st.subheader("Suggestions")
        if suggestions.empty:
            st.info("No suggestions found")
        else:
            st.dataframe(
                suggestions,
                width="stretch",
                column_config={
                    "Personal webpage": st.column_config.LinkColumn(
                        display_text=r"https?://(.*)$"
                    )
                },
            )
