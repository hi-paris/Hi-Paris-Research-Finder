import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import ast


# ROOT DIR pour import backend
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.exact_matching import exact_match
from backend.semantic_search import semantic_search_filtered

# Page configuration
st.set_page_config(
    page_title="Hi! Paris Research Profiles",
    page_icon=r"logo/icon_hi_search.png",
    layout="wide",
)

# Load data
file_path = "data/prof_ecosystem_01_21.xlsx"
df = pd.read_excel(file_path)

# Processing
df["Research domains"] = df["Research domains"].apply(ast.literal_eval)
affiliations = df["Affiliation"].dropna().unique().tolist()

# Title
st.title("Profile Finder")
st.markdown("""
    Enter a research domain below to find all relevant researcher profiles affiliated with Hi! Paris.
    """)

# Sidebar options
st.sidebar.image(r"logo/logo_hi_paris.png", width=300)
st.sidebar.header("Options")
show_exact = st.sidebar.checkbox("Show exact matches", value=True)
show_suggestions = st.sidebar.checkbox("Show suggestions", value=False)

# Affiliation filter
selected_affiliations = st.sidebar.multiselect(
    "Filter by affiliation:", options=affiliations, default=None
)

# Match Level explanation
st.sidebar.markdown("### Match Level for Suggestions")
st.sidebar.markdown("""
- **Excellent**: Works closely on this topic  
- **Good**: Works often on this topic  
- **Fair**: Works sometimes on this topic  
- **Low**: Works little on this topic  
""")


# User input
query = st.text_input(
    "Enter a research domain:",
    placeholder="e.g. optimal transport, time series, computer vision",
)


# Main
# Main
if not query:
    # Case 1: No research domain, but affiliation selected
    if selected_affiliations:
        df_filtered = df[df["Affiliation"].isin(selected_affiliations)]

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

    # Optional: no query & no affiliation
    else:
        st.info("Please enter a research domain or select an affiliation.")

else:
    # Existing behavior (unchanged)
    if selected_affiliations:
        df_filtered = df[df["Affiliation"].isin(selected_affiliations)]
    else:
        df_filtered = df

    # Exact match
    exact_results = exact_match(query, df_filtered)
    exact_ids = set(exact_results["ID"].tolist()) if not exact_results.empty else set()

    if show_exact:
        if not exact_results.empty:
            st.subheader("Exact matches")
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
        else:
            st.info("No exact match found")

    # Suggestions
    if show_suggestions:
        suggestions = semantic_search_filtered(
            query=query,
            df=df_filtered,
            exact_ids=exact_ids,
            top_k=10,
            threshold=0.4,
        )

        if not suggestions.empty:
            st.subheader("Suggestions")
            st.dataframe(suggestions, width="stretch")
        else:
            st.info("No suggestions found")
