"""
Regenerate Manna.pkl — compatible with pandas 2.2.x / numpy 2.x
Author: Madhusudan Manna
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

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
chunks = []
for chunk in pd.read_csv('zomato.csv', encoding='latin-1',
                         on_bad_lines='skip', low_memory=True,
                         chunksize=10000):
    chunks.append(chunk[NEEDED])

df = pd.concat(chunks, ignore_index=True)
df.columns = ['name', 'cuisines', 'rate', 'reviews_list',
              'rest_type', 'location', 'cost', 'votes']
print(f"Loaded raw: {df.shape}")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
df.drop_duplicates(subset=['name'], keep='first', inplace=True)
df.dropna(subset=['name', 'reviews_list'], inplace=True)

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
df['cost'] = pd.to_numeric(
    df['cost'].astype(str).str.replace(',', '', regex=False),
    errors='coerce'
)
df.reset_index(drop=True, inplace=True)
print(f"After cleaning: {df.shape}")

# ── 3. Force all object columns to plain Python str (avoid StringDtype issues) ─
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# ── 4. Parse review text ──────────────────────────────────────────────────────
def extract_review_text(raw):
    try:
        parts = re.findall(r"'(.*?)'", str(raw), re.DOTALL)
        texts = [p for p in parts if len(p) > 20]
        return ' '.join(texts)
    except:
        return ''

print("Parsing review text...")
df['parsed_reviews'] = df['reviews_list'].apply(extract_review_text)

# ── 5. Clean text ─────────────────────────────────────────────────────────────
print("Cleaning text...")
df['cleaned_reviews'] = (
    df['parsed_reviews']
    .str.lower()
    .str.replace(r'rated\s*\d[\d\.]*', ' ', regex=True)
    .str.replace(r'&nbsp;', ' ', regex=False)
    .str.replace(r'[^a-z\s]', ' ', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)
df = df[df['cleaned_reviews'].str.len() > 30].reset_index(drop=True)
print(f"Valid review rows: {len(df)}")

# ── 6. TF-IDF ─────────────────────────────────────────────────────────────────
print("Fitting TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 1),
    stop_words='english',
    min_df=2,
    dtype=np.float32
)
tfidf_matrix = tfidf.fit_transform(df['cleaned_reviews'])
print(f"TF-IDF matrix: {tfidf_matrix.shape}")

# ── 7. Cosine Similarity (batched) ───────────────────────────────────────────
print("Computing cosine similarity...")
n = tfidf_matrix.shape[0]
BATCH = 2000
cosine_sim = np.zeros((n, n), dtype=np.float32)

for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    cosine_sim[start:end] = cosine_similarity(
        tfidf_matrix[start:end], tfidf_matrix)
    print(f"  {end}/{n} ({int(end/n*100)}%)")

print(f"Similarity matrix: {cosine_sim.shape}")

# ── 8. Index series ───────────────────────────────────────────────────────────
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

# ── 9. Keep only essential columns and ensure plain dtypes ───────────────────
df_save = df[['name', 'cuisines', 'rate', 'rest_type',
              'location', 'cost', 'votes']].copy()

# Convert every column to native Python types — no pandas ExtensionDtype
for col in df_save.select_dtypes(include='object').columns:
    df_save[col] = df_save[col].astype(str)
df_save['rate']  = df_save['rate'].astype(float)
df_save['votes'] = pd.to_numeric(df_save['votes'], errors='coerce').fillna(0).astype(int)
df_save['cost']  = df_save['cost'].astype(float)

# ── 10. Save ──────────────────────────────────────────────────────────────────
print("Saving Manna.pkl ...")
model_data = {
    'df'        : df_save,
    'cosine_sim': cosine_sim,
    'indices'   : indices
}
with open('Manna.pkl', 'wb') as f:
    pickle.dump(model_data, f, protocol=4)   # protocol 4 = Python 3.4+ safe

# Quick verify
with open('Manna.pkl', 'rb') as f:
    verify = pickle.load(f)
print("=" * 50)
print("✅  Manna.pkl saved and verified!")
print(f"   Restaurants  : {len(verify['df'])}")
print(f"   Similarity   : {verify['cosine_sim'].shape}")
print(f"   Index entries: {len(verify['indices'])}")
print("=" * 50)


