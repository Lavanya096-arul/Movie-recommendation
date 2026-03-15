# ===============================
# 🎬 MOVIE RECOMMENDATION SYSTEM - STREAMLIT (OPTION A)
# Improved ML Quality
# ===============================

import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# Load & cache dataset
# -------------------------------
@st.cache_data
def load_data():
    # Load movies
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]
    
    # Load credits
    credits = pd.read_csv("tmdb_5000_credits.csv")
    # Replace 'movie_id' with 'id' to match movies.csv
    if 'movie_id' in credits.columns:
        credits.rename(columns={'movie_id': 'id'}, inplace=True)
    
    credits = credits[['id', 'cast', 'crew']]
    
    # Merge datasets on 'id'
    df = movies.merge(credits, on='id')
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# -------------------------------
# Helper functions
# -------------------------------
def extract_names(text, limit=None):
    try:
        items = ast.literal_eval(text)
        names = [i['name'] for i in items]
        return " ".join(names[:limit]) if limit else " ".join(names)
    except:
        return ""

def extract_director(text):
    try:
        crew = ast.literal_eval(text)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return ""
    except:
        return ""

def clean_text(text):
    if isinstance(text, str):
        return text.lower().replace(",", "").replace(".", "")
    return ""

# -------------------------------
# Build ML model (cached)
# -------------------------------
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=8000,
        ngram_range=(1, 2)
    )
    vectors = tfidf.fit_transform(df['tags'])
    similarity = cosine_similarity(vectors)
    return similarity

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie_name):
    if movie_name not in df['title'].values:
        return pd.DataFrame()

    index = df[df['title'] == movie_name].index[0]
    distances = similarity[index]

    movies = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    return df.iloc[[i[0] for i in movies]]

# ===============================
# MAIN APP
# ===============================

# Load data
df = load_data()

# -------------------------------
# Feature engineering (OPTION A MAGIC ✨)
# -------------------------------
df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(lambda x: extract_names(x, limit=3))
df['director'] = df['crew'].apply(extract_director)
df['overview'] = df['overview'].apply(clean_text)

df['tags'] = (
    df['genres'] + " " +
    df['keywords'] + " " +
    df['cast'] + " " +
    df['director'] + " " +
    df['overview']
)

# Build similarity matrix
similarity = build_model(df)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🎥 Movie Recommender")
search = st.sidebar.text_input("🔍 Search movie")

filtered_movies = (
    df[df['title'].str.contains(search, case=False)]
    if search else df
)

selected_movie = st.sidebar.selectbox(
    "🎬 Choose a movie",
    filtered_movies['title'].values
)

# -------------------------------
# Main UI
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Top **5 similar movies** using enhanced Machine Learning ✨")

if st.button("🔥 Show Recommendations"):
    with st.spinner("Finding similar movies 🍿..."):
        recommendations = recommend(selected_movie)

    if not recommendations.empty:
        cols = st.columns(5)
        for i, movie in enumerate(recommendations.itertuples()):
            with cols[i]:
                st.markdown(f"### {movie.title}")
                st.caption(movie.overview[:120] + "...")
    else:
        st.warning("Movie not found in the dataset!")
