"""
Zomato Restaurant Recommendation System
Author: Madhusudan Manna
Streamlit Web Application
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #E23744;
            text-align: center;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        .author {
            text-align: center;
            color: #888;
            font-style: italic;
            margin-bottom: 20px;
        }
        .recommend-header {
            color: #E23744;
            font-size: 1.4rem;
            font-weight: bold;
        }
        .stDataFrame { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🍽️ Zomato Restaurant Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Content-Based Recommender using NLP & Cosine Similarity</div>', unsafe_allow_html=True)
st.markdown('<div class="author">By: Madhusudan Manna</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Load Pickle Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load Manna.pkl.
    Uses protocol=4 compatible pickle. Returns (df, cosine_sim, indices).
    """
    import os
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Manna.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    df        = data['df']
    cosine_sim = data['cosine_sim']
    indices   = data['indices']
    # Ensure all string columns are plain str, not pandas StringDtype
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df, cosine_sim, indices

try:
    df, cosine_sim, indices = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ 'Manna.pkl' not found. Please run `generate_pickle.py` first.")
    st.stop()
except TypeError as e:
    st.error(f"❌ Pickle version mismatch: {e}\n\nPlease re-run `generate_pickle.py` to regenerate Manna.pkl.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {type(e).__name__}: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Recommendation Function
# ─────────────────────────────────────────────
def recommend_restaurants(restaurant_name, top_n=10):
    """
    Recommends top N restaurants similar to the given restaurant,
    sorted by highest rating.
    """
    name_lower = restaurant_name.lower()

    if name_lower not in indices:
        return None

    idx = indices[name_lower]

    # Similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    restaurant_indices = [i[0] for i in sim_scores]

    result = df.iloc[restaurant_indices][
        ['name', 'cuisines', 'rate', 'rest_type', 'location', 'cost']
    ].copy()
    result['similarity_score'] = [round(i[1], 4) for i in sim_scores]

    # Sort by rating descending
    result = result.sort_values(by='rate', ascending=False)
    result.reset_index(drop=True, inplace=True)
    result.index += 1  # Start from 1
    return result

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png", width=200)
st.sidebar.markdown("## 🔧 Settings")
top_n = st.sidebar.slider("Number of Recommendations", min_value=5, max_value=20, value=10, step=1)
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app recommends restaurants based on **review similarity** using "
    "TF-IDF vectorization and Cosine Similarity. Results are sorted by "
    "**highest rating** first."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Madhusudan Manna")
st.sidebar.markdown("**Tech:** Python · NLP · Streamlit · Scikit-learn")

# ─────────────────────────────────────────────
# Main Input Section
# ─────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    all_restaurants = sorted(df['name'].dropna().unique().tolist())
    restaurant_input = st.selectbox(
        "🔍 Select or type a Restaurant Name:",
        options=all_restaurants,
        index=0
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🍴 Get Recommendations", use_container_width=True)

# ─────────────────────────────────────────────
# Show Results
# ─────────────────────────────────────────────
if search_btn:
    if restaurant_input:
        with st.spinner(f"Finding restaurants similar to **{restaurant_input}**..."):
            result = recommend_restaurants(restaurant_input, top_n=top_n)

        if result is None:
            st.warning(f"⚠️ Restaurant **'{restaurant_input}'** not found. Please try another name.")
        else:
            st.markdown(f"### 🎯 Top {top_n} Recommendations for: `{restaurant_input}`")
            st.markdown("*Sorted by Highest Rating ⭐*")
            st.markdown("---")

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("🏆 Top Rated", f"{result['rate'].max():.1f} / 5.0")
            m2.metric("📊 Avg Rating", f"{result['rate'].mean():.2f} / 5.0")
            m3.metric("🍽️ Restaurants Found", len(result))

            st.markdown("---")

            # Display DataFrame
            st.dataframe(
                result.rename(columns={
                    'name': 'Restaurant Name',
                    'cuisines': 'Cuisines',
                    'rate': 'Rating ⭐',
                    'rest_type': 'Type',
                    'location': 'Location',
                    'cost': 'Cost for 2 (₹)',
                    'similarity_score': 'Similarity Score'
                }),
                use_container_width=True,
                height=420
            )

            # Bar chart of ratings
            st.markdown("### 📊 Rating Comparison")
            chart_data = result[['name', 'rate']].set_index('name')
            st.bar_chart(chart_data)
    else:
        st.warning("Please select or type a restaurant name.")

# ─────────────────────────────────────────────
# Dataset Overview
# ─────────────────────────────────────────────
with st.expander("📂 View Dataset Overview"):
    st.write(f"**Total Restaurants:** {len(df)}")
    st.write(f"**Average Rating:** {df['rate'].mean():.2f}")
    st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by <b>Madhusudan Manna</b> | "
    "Zomato Restaurant Recommendation System</center>",
    unsafe_allow_html=True
)
