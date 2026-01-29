# backend/matching.py
import pandas as pd

# backend/matching.py
import pandas as pd


def normalize(text):
    if pd.isna(text):
        return None
    return text.strip().lower()



def exact_match(query, df):
    q = query.strip().lower()

    matched_df = df[
        df["Research domains"].apply(
            lambda ds: any(q in d.lower().strip() or d.lower().strip() in q for d in ds)
        )
    ]

    return matched_df.reset_index(drop=True)