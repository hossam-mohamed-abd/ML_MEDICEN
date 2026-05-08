from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from difflib import get_close_matches

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "clean_ml_dataset_fixed.csv"
)

df = df.fillna("")

# lowercase

text_columns = [
    "drug_name",
    "active_substance",
    "dosage_form",
    "indication",
    "features"
]

for col in text_columns:

    df[col] = (

        df[col]

        .astype(str)

        .str.lower()

        .str.strip()
    )

# =====================================================
# TF-IDF
# =====================================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    df["features"]
)

# =====================================================
# COSINE SIMILARITY
# =====================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

# =====================================================
# ARABIC TRANSLITERATION
# =====================================================

arabic_map = {

    "ا": "a",
    "أ": "a",
    "إ": "e",
    "آ": "a",

    "ب": "b",

    "ت": "t",

    "ث": "th",

    "ج": "g",

    "ح": "h",

    "خ": "kh",

    "د": "d",

    "ذ": "z",

    "ر": "r",

    "ز": "z",

    "س": "s",

    "ش": "sh",

    "ص": "s",

    "ض": "d",

    "ط": "t",

    "ظ": "z",

    "ع": "a",

    "غ": "gh",

    "ف": "f",

    "ق": "q",

    "ك": "k",

    "ل": "l",

    "م": "m",

    "ن": "n",

    "ه": "h",

    "و": "o",

    "ي": "y",

    "ى": "a"
}

# =====================================================
# TRANSLITERATION FUNCTION
# =====================================================

def transliterate_arabic(text):

    result = ""

    for char in text:

        if char in arabic_map:

            result += arabic_map[char]

        else:

            result += char

    return result

# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================

def recommend_drugs(drug_name, top_n=5):

    drug_name = drug_name.lower().strip()

    transliterated_name = transliterate_arabic(
        drug_name
    )

    all_drugs = df["drug_name"].dropna().unique()

    close_matches = get_close_matches(

        transliterated_name,

        all_drugs,

        n=5,

        cutoff=0.4
    )

    if len(close_matches) == 0:

        return []

    drug_name = close_matches[0]

    matches = df[
        df["drug_name"] == drug_name
    ]

    idx = matches.index[0]

    scores = list(
        enumerate(similarity_matrix[idx])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:]

    recommended_names = set()

    filtered_scores = []

    for i, score in scores:

        recommended_drug = df.iloc[i]["drug_name"]

        if recommended_drug == drug_name:
            continue

        if recommended_drug in recommended_names:
            continue

        if score < 0.25:
            continue

        recommended_names.add(
            recommended_drug
        )

        filtered_scores.append({

            "drug_name":
            df.iloc[i]["drug_name"],

            "active_substance":
            df.iloc[i]["active_substance"],

            "indication":
            df.iloc[i]["indication"],

            "similarity_score":
            round(float(score), 3)
        })

        if len(filtered_scores) >= top_n:
            break

    return filtered_scores

# =====================================================
# HOME ROUTE
# =====================================================

@app.get("/")

def home():

    return {
        "message":
        "Drug Recommendation API Running"
    }

# =====================================================
# RECOMMENDATION ROUTE
# =====================================================

@app.get("/recommend/{drug_name}")

def recommend(drug_name: str):

    results = recommend_drugs(drug_name)

    return {
        "searched_drug": drug_name,
        "recommendations": results
    }