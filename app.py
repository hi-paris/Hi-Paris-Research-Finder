import ast
import pandas as pd
import streamlit as st

from utils import file_hash, apply_filters
from search import (
    load_or_build_index,
    exact_match,
    semantic_search_fast,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Hi! Paris Research Profiles",
    page_icon=r"logo/icon_hi_search.png",
    layout="wide",
)

# =============================================================================
# AUTH
# =============================================================================
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter password",
            type="password",
            key="password",
            on_change=password_entered,
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter password",
            type="password",
            key="password",
            on_change=password_entered,
        )
        st.error("Incorrect password")
        return False
    else:
        return True


if not check_password():
    st.stop()


# =============================================================================
# DATA
# =============================================================================
csv_url = st.secrets["google_sheet_url"]
df = pd.read_csv(csv_url)
excel_hash = file_hash(df)

df["Research domains"] = df["Research domains"].apply(ast.literal_eval)

affiliations = df["Affiliation"].dropna().unique().tolist()
research_axis_options = df["Research axis"].dropna().unique().tolist()

model, index, df_copy, ids = load_or_build_index(df, excel_hash)


# =============================================================================
# UI
# =============================================================================
st.title("Hi! Paris Expert Finder")

st.sidebar.image(r"logo/logo_hi_paris.png", width=300)
st.sidebar.header("Options")
show_exact = st.sidebar.checkbox("Show exact matches", value=True)
show_suggestions = st.sidebar.checkbox("Show suggestions", value=True)

query = st.text_input("Enter a research domain:")

col1, col2 = st.columns(2)

with col1:
    selected_affiliations = st.multiselect(
        "Filter by affiliation:", options=affiliations
    )

with col2:
    selected_research_axis = st.multiselect(
        "Filter by Research axis:", options=research_axis_options
    )


# =============================================================================
# LOGIC
# =============================================================================
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
            model,
            index,
            df_copy,
            ids,
            exact_ids=exact_ids,
            allowed_axes=selected_research_axis,
            allowed_affiliations=selected_affiliations

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