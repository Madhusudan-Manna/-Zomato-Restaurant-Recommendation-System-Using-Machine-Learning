"""
Zomato Restaurant Recommendation System
Author : Madhusudan Manna
Tech   : Python · NLP · TF-IDF · Cosine Similarity · Streamlit
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:700; color:#E23744; text-align:center; }
    .sub-title   { font-size:1.05rem; color:#555; text-align:center; margin-bottom:8px; }
    .author-tag  { text-align:center; color:#888; font-style:italic; margin-bottom:10px; }
    .stDataFrame { border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🍽️ Zomato Restaurant Recommendation System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Content-Based Recommender · NLP · TF-IDF · Cosine Similarity</div>',
            unsafe_allow_html=True)
st.markdown('<div class="author-tag">By: Madhusudan Manna</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Load model  — Manna.pkl contains: df, cosine_sim, indices
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Manna.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df         = data["df"]
    cosine_sim = data["cosine_sim"]
    indices    = data["indices"]

    # Ensure plain object dtype — avoids pandas 2.2 StringDtype issues
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    return df, cosine_sim, indices


try:
    df, cosine_sim, indices = load_model()
    st.success(f"✅ Model loaded — {len(df):,} restaurants indexed.")
except FileNotFoundError:
    st.error("❌ Manna.pkl not found. Run `generate_pickle.py` first.")
    st.stop()
except KeyError as e:
    st.error(f"❌ Manna.pkl is missing key: {e}. Re-run `generate_pickle.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model — {type(e).__name__}: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Recommendation function
# ─────────────────────────────────────────────────────────────────────────────
def recommend(restaurant_name: str, top_n: int = 10) -> pd.DataFrame | None:
    """
    Returns top_n restaurants most similar to restaurant_name,
    sorted by rating descending.
    Uses the precomputed cosine_sim matrix from Manna.pkl.
    """
    key = restaurant_name.strip().lower()

    if key not in indices:
        return None

    idx = indices[key]

    # cosine_sim row for this restaurant
    scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity descending; skip index 0 (the restaurant itself)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

    row_ids = [s[0] for s in scores]

    cols = ["name", "cuisines", "rate", "rest_type", "location", "cost"]
    # use only columns that exist in df
    cols = [c for c in cols if c in df.columns]

    result = df.iloc[row_ids][cols].copy()
    result["similarity_score"] = [round(s[1], 4) for s in scores]

    # Sort by highest rating first
    result = result.sort_values("rate", ascending=False).reset_index(drop=True)
    result.index += 1
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png",
    width=190
)
st.sidebar.markdown("## ⚙️ Settings")
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10, 1)
st.sidebar.markdown("---")
st.sidebar.info(
    "Recommends restaurants with **similar review content** using "
    "TF-IDF + Cosine Similarity, sorted by **highest rating first**."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Madhusudan Manna  \n**Tech:** Python · NLP · Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
all_names = sorted(df["name"].dropna().unique().tolist())

col1, col2 = st.columns([4, 1])
with col1:
    restaurant_input = st.selectbox(
        "🔍 Select or type a Restaurant Name:",
        options=all_names,
        index=0
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🍴 Recommend", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if search_btn:
    with st.spinner(f"Finding restaurants similar to **{restaurant_input}**…"):
        result = recommend(restaurant_input, top_n=top_n)

    if result is None:
        st.warning(
            f"⚠️ **'{restaurant_input}'** not found in the dataset. "
            "Try selecting a different name from the dropdown."
        )
    else:
        st.markdown(f"### 🎯 Top {top_n} Recommendations for: `{restaurant_input}`")
        st.caption("Sorted by highest rating ⭐")
        st.markdown("---")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 Top Rated",       f"{result['rate'].max():.1f} / 5.0")
        m2.metric("📊 Average Rating",  f"{result['rate'].mean():.2f} / 5.0")
        m3.metric("🍽️ Results",         str(len(result)))
        m4.metric("🔗 Best Similarity", f"{result['similarity_score'].max():.4f}")

        st.markdown("---")

        # Rename columns for display
        display_cols = {
            "name":             "Restaurant Name",
            "cuisines":         "Cuisines",
            "rate":             "Rating ⭐",
            "rest_type":        "Type",
            "location":         "Location",
            "cost":             "Cost for 2 (₹)",
            "similarity_score": "Similarity Score",
        }
        st.dataframe(
            result.rename(columns=display_cols),
            use_container_width=True,
            height=400
        )

        # Bar chart
        st.markdown("### 📊 Rating Comparison")
        chart = result[["name", "rate"]].set_index("name")
        st.bar_chart(chart)

        # Cuisine breakdown
        if "cuisines" in result.columns:
            st.markdown("### 🍜 Cuisine Breakdown")
            cuisine_counts = (
                result["cuisines"]
                .str.split(",")
                .explode()
                .str.strip()
                .value_counts()
                .head(10)
            )
            st.bar_chart(cuisine_counts)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset overview expander
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📂 Dataset Overview"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Restaurants", f"{len(df):,}")
    c2.metric("Average Rating",    f"{df['rate'].mean():.2f}")
    c3.metric("Locations",         str(df['location'].nunique()) if 'location' in df.columns else "—")
    st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by <b>Madhusudan Manna</b> | "
    "Zomato Restaurant Recommendation System</center>",
    unsafe_allow_html=True
)
