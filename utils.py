import hashlib


def file_hash(data):
    """
    Hash function used to detect whether we need to rebuild the FAISS index.
    """
    csv_bytes = data.to_csv(index=False).encode()
    return hashlib.md5(csv_bytes).hexdigest()


def build_prof_text(row):
    domains = row["Research domains"]
    domains_text = ", ".join(domains) if isinstance(domains, list) else str(domains)
    return (
        f"passage: Research domains fields: {domains_text}. "
        f"Summary of the professor: {row.get('Summary', '')}"
    )


def label_match(score):
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"


def apply_filters(df, affiliations, research_axes):
    if affiliations:
        df = df[df["Affiliation"].isin(affiliations)]
    if research_axes:
        df = df[df["Research axis"].isin(research_axes)]
    return df