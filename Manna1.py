"""
Generate Manna1.pkl  — lightweight version for cloud deployment (< 15 MB)
Strategy : store the sparse TF-IDF matrix instead of the dense 130 MB
           cosine similarity matrix.  Similarity is computed on-the-fly
           for ONE query restaurant at a time (~0.01 sec per query).
Author   : Madhusudan Manna
"""

import numpy as np
import pandas as pd
import re, pickle, warnings, os
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz
import scipy.sparse as sp

# ── Step 1: Load CSV ──────────────────────────────────────────────────────────
NEEDED = ['name','cuisines','rate','reviews_list',
          'rest_type','location','approx_cost(for two people)','votes']

print("Step 1: Loading dataset...")
chunks = []
for chunk in pd.read_csv('zomato.csv', encoding='latin-1',
                         on_bad_lines='skip', low_memory=True,
                         chunksize=10000):
    chunks.append(chunk[NEEDED])

df = pd.concat(chunks, ignore_index=True)
df.columns = ['name','cuisines','rate','reviews_list',
              'rest_type','location','cost','votes']
print(f"  Loaded raw: {df.shape}")

# ── Step 2: Deduplicate & drop nulls ─────────────────────────────────────────
df.drop_duplicates(subset=['name'], keep='first', inplace=True)
df.dropna(subset=['name','reviews_list'], inplace=True)

# ── Step 3: Clean rate ────────────────────────────────────────────────────────
def clean_rate(r):
    try:
        r = str(r).strip()
        if r in ['NEW','-','nan','']: return np.nan
        return float(r.split('/')[0].strip())
    except: return np.nan

df['rate'] = df['rate'].apply(clean_rate)
df.dropna(subset=['rate'], inplace=True)

# ── Step 4: Clean cost ────────────────────────────────────────────────────────
df['cost'] = pd.to_numeric(
    df['cost'].astype(str).str.replace(',','',regex=False), errors='coerce')
df.reset_index(drop=True, inplace=True)
print(f"  After cleaning: {df.shape}")

# ── Step 5: Plain string dtypes ───────────────────────────────────────────────
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# ── Step 6: Parse review text ─────────────────────────────────────────────────
print("Step 6: Parsing reviews...")
def extract_review_text(raw):
    try:
        parts = re.findall(r"'(.*?)'", str(raw), re.DOTALL)
        return ' '.join(p for p in parts if len(p) > 20)
    except: return ''

df['parsed_reviews'] = df['reviews_list'].apply(extract_review_text)

# ── Step 7: Clean text ────────────────────────────────────────────────────────
print("Step 7: Cleaning text...")
df['cleaned_reviews'] = (
    df['parsed_reviews']
    .str.lower()
    .str.replace(r'rated\s*\d[\d\.]*',' ', regex=True)
    .str.replace(r'&nbsp;',' ', regex=False)
    .str.replace(r'[^a-z\s]',' ', regex=True)
    .str.replace(r'\s+',' ', regex=True)
    .str.strip()
)
df = df[df['cleaned_reviews'].str.len() > 30].reset_index(drop=True)
print(f"  Valid rows: {len(df)}")

# ── Step 8: TF-IDF (sparse) ───────────────────────────────────────────────────
print("Step 8: TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=3000, ngram_range=(1,1),
    stop_words='english', min_df=2, dtype=np.float32
)
tfidf_matrix = tfidf.fit_transform(df['cleaned_reviews'])  # sparse CSR float32
print(f"  Matrix: {tfidf_matrix.shape}  nnz={tfidf_matrix.nnz:,}")
sparse_mb = tfidf_matrix.data.nbytes / 1024 / 1024
print(f"  Sparse data size: {sparse_mb:.2f} MB")

# ── Step 9: Index series ──────────────────────────────────────────────────────
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

# ── Step 10: Clean save DataFrame ────────────────────────────────────────────
df_save = df[['name','cuisines','rate','rest_type','location','cost','votes']].copy()
for col in df_save.select_dtypes(include='object').columns:
    df_save[col] = df_save[col].astype(str)
df_save['rate']  = df_save['rate'].astype(float)
df_save['cost']  = df_save['cost'].astype(float)
df_save['votes'] = pd.to_numeric(df_save['votes'], errors='coerce').fillna(0).astype(int)

# ── Step 11: Save Manna1.pkl ──────────────────────────────────────────────────
# Store sparse tfidf_matrix (not the dense cosine matrix)
print("Step 11: Saving Manna1.pkl...")
model_data = {
    'df'          : df_save,
    'tfidf_matrix': tfidf_matrix,   # scipy sparse CSR — tiny
    'indices'     : indices
}
with open('Manna1.pkl', 'wb') as f:
    pickle.dump(model_data, f, protocol=4)

# ── Step 12: Verify ───────────────────────────────────────────────────────────
with open('Manna1.pkl', 'rb') as f:
    v = pickle.load(f)

size_mb = os.path.getsize('Manna1.pkl') / 1024 / 1024

# Quick smoke-test: recommend for first restaurant
def quick_recommend(name, top_n=5):
    key = name.strip().lower()
    if key not in v['indices']: return 'NOT FOUND'
    idx = v['indices'][key]
    row = v['tfidf_matrix'][idx]                         # 1×3000 sparse
    scores = cosine_similarity(row, v['tfidf_matrix']).flatten()
    top = scores.argsort()[::-1][1:top_n+1]
    result = v['df'].iloc[top][['name','rate']].copy()
    result['score'] = scores[top].round(4)
    return result.sort_values('rate', ascending=False).reset_index(drop=True)

test_name = v['df']['name'].iloc[0]
rec = quick_recommend(test_name)

print("=" * 55)
print("✅  Manna1.pkl saved and verified!")
print(f"   File size    : {size_mb:.2f} MB")
print(f"   Restaurants  : {len(v['df'])}")
print(f"   TF-IDF shape : {v['tfidf_matrix'].shape}")
print(f"   Index entries: {len(v['indices'])}")
print(f"   Test query   : '{test_name}'")
print(f"   Recommendations:")
print(rec.to_string(index=False))
print("=" * 55)
print()
if size_mb <= 15:
    print(f"   ✅ Under 15 MB — ready for cloud deployment!")
else:
    print(f"   ⚠️  {size_mb:.1f} MB — above 15 MB target")
