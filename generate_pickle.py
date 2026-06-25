"""
Generate Manna.pkl — Zomato Restaurant Recommendation System
Compatible with pandas 2.2.x / numpy 2.x / Python 3.9+
Author: Madhusudan Manna

Saves: df, cosine_sim, indices
"""

import numpy as np
import pandas as pd
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEEDED = ['name', 'cuisines', 'rate', 'reviews_list',
          'rest_type', 'location', 'approx_cost(for two people)', 'votes']

# ── Step 1: Load CSV in chunks (handles 547 MB file safely) ──────────────────
print("Step 1: Loading dataset...")
chunks = []
for chunk in pd.read_csv(
        'zomato.csv', encoding='latin-1',
        on_bad_lines='skip', low_memory=True,
        chunksize=10000):
    chunks.append(chunk[NEEDED])

df = pd.concat(chunks, ignore_index=True)
df.columns = ['name', 'cuisines', 'rate', 'reviews_list',
              'rest_type', 'location', 'cost', 'votes']
print(f"  Loaded raw: {df.shape}")

# ── Step 2: Deduplicate and drop nulls ────────────────────────────────────────
print("Step 2: Deduplicating and dropping nulls...")
df.drop_duplicates(subset=['name'], keep='first', inplace=True)
df.dropna(subset=['name', 'reviews_list'], inplace=True)

# ── Step 3: Clean rate column (e.g. '4.1/5' → 4.1, 'NEW' → NaN) ─────────────
print("Step 3: Cleaning rate column...")
def clean_rate(r):
    try:
        r = str(r).strip()
        if r in ['NEW', '-', 'nan', '']:
            return np.nan
        return float(r.split('/')[0].strip())
    except:
        return np.nan

df['rate'] = df['rate'].apply(clean_rate)
df.dropna(subset=['rate'], inplace=True)

# ── Step 4: Clean cost column ─────────────────────────────────────────────────
print("Step 4: Cleaning cost column...")
df['cost'] = pd.to_numeric(
    df['cost'].astype(str).str.replace(',', '', regex=False),
    errors='coerce'
)
df.reset_index(drop=True, inplace=True)
print(f"  After cleaning: {df.shape}")

# ── Step 5: Force all string columns to plain object dtype ────────────────────
# Prevents pandas 2.2 StringDtype incompatibility in pickle
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# ── Step 6: Parse review text from stringified list-of-tuples ────────────────
# reviews_list looks like: "[('Rated 4.0', 'RATED\n  Great food...'), ...]"
print("Step 6: Parsing review text...")
def extract_review_text(raw):
    try:
        parts = re.findall(r"'(.*?)'", str(raw), re.DOTALL)
        texts = [p for p in parts if len(p) > 20]
        return ' '.join(texts)
    except:
        return ''

df['parsed_reviews'] = df['reviews_list'].apply(extract_review_text)

# ── Step 7: Clean text (vectorised — no slow row-by-row stemming) ─────────────
print("Step 7: Cleaning review text...")
df['cleaned_reviews'] = (
    df['parsed_reviews']
    .str.lower()
    .str.replace(r'rated\s*\d[\d\.]*', ' ', regex=True)  # remove 'rated 4.0'
    .str.replace(r'&nbsp;',            ' ', regex=False)  # remove HTML entities
    .str.replace(r'[^a-z\s]',         ' ', regex=True)   # keep letters only
    .str.replace(r'\s+',              ' ', regex=True)   # collapse spaces
    .str.strip()
)
# Drop rows where cleaned review is too short to be useful
df = df[df['cleaned_reviews'].str.len() > 30].reset_index(drop=True)
print(f"  Valid review rows: {len(df)}")

# ── Step 8: TF-IDF Vectorization ──────────────────────────────────────────────
print("Step 8: Fitting TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 1),
    stop_words='english',
    min_df=2,
    dtype=np.float32
)
tfidf_matrix = tfidf.fit_transform(df['cleaned_reviews'])
print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

# ── Step 9: Cosine Similarity (batched to avoid out-of-memory) ───────────────
print("Step 9: Computing cosine similarity in batches...")
n = tfidf_matrix.shape[0]
BATCH = 2000
cosine_sim = np.zeros((n, n), dtype=np.float32)

for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    cosine_sim[start:end] = cosine_similarity(
        tfidf_matrix[start:end], tfidf_matrix)
    print(f"  {end}/{n} ({int(end / n * 100)}%)")

print(f"  Similarity matrix: {cosine_sim.shape}")

# ── Step 10: Build name → index lookup series ─────────────────────────────────
print("Step 10: Building index series...")
indices = pd.Series(
    df.index,
    index=df['name'].str.lower()
).drop_duplicates()
print(f"  Index entries: {len(indices)}")

# ── Step 11: Prepare clean save DataFrame ────────────────────────────────────
print("Step 11: Preparing save DataFrame...")
df_save = df[['name', 'cuisines', 'rate', 'rest_type',
              'location', 'cost', 'votes']].copy()

# Ensure native Python types — no pandas ExtensionDtype in pickle
for col in df_save.select_dtypes(include='object').columns:
    df_save[col] = df_save[col].astype(str)
df_save['rate']  = df_save['rate'].astype(float)
df_save['cost']  = df_save['cost'].astype(float)
df_save['votes'] = pd.to_numeric(
    df_save['votes'], errors='coerce').fillna(0).astype(int)

# ── Step 12: Save Manna.pkl ───────────────────────────────────────────────────
print("Step 12: Saving Manna.pkl...")
model_data = {
    'df'        : df_save,
    'cosine_sim': cosine_sim,
    'indices'   : indices
}
with open('Manna.pkl', 'wb') as f:
    pickle.dump(model_data, f, protocol=4)   # protocol 4 = Python 3.4+ safe

# ── Step 13: Verify ───────────────────────────────────────────────────────────
print("Step 13: Verifying Manna.pkl...")
with open('Manna.pkl', 'rb') as f:
    verify = pickle.load(f)

import os
size_mb = os.path.getsize('Manna.pkl') / (1024 * 1024)

print("=" * 50)
print("✅  Manna.pkl saved and verified!")
print(f"   File size    : {size_mb:.1f} MB")
print(f"   Restaurants  : {len(verify['df'])}")
print(f"   Similarity   : {verify['cosine_sim'].shape}")
print(f"   Index entries: {len(verify['indices'])}")
print(f"   df dtypes    : {dict(verify['df'].dtypes)}")
print("=" * 50)
print()
print("Run the app with:  streamlit run app.py")
