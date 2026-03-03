import hashlib
from collections.abc import Sequence
import pandas as pd


def file_hash(data: pd.DataFrame) -> str:
    """
    Compute a deterministic MD5 hash of a pandas DataFrame.
    This hash is used to detect whether the dataset has changed.
    """
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    return hashlib.md5(csv_bytes).hexdigest()


def build_prof_text(row: pd.Series) -> str:
    """
    Build a formatted textual representation of a professor's profile.
    """
    domains = row["Research domains"]
    domains_text = ", ".join(domains) if isinstance(domains, list) else str(domains)

    return (
        f"passage: Research domains fields: {domains_text}. "
        f"Summary of the professor: {row.get('Summary', '')}"
    )


def label_match(score: float) -> str:
    """
    Categorize a similarity or relevance score into a qualitative label.
    """
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"


def apply_filters(
    df: pd.DataFrame,
    affiliations: Sequence[str] | None = None,
    research_axes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Filter a DataFrame of professors based on affiliations and research axes.
    """
    if affiliations:
        df = df[df["Affiliation"].isin(affiliations)]
    if research_axes:
        df = df[df["Research axis"].isin(research_axes)]
    return df
