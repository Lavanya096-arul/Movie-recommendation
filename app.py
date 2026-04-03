import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['id','title','overview','genres','keywords']]

    credits = pd.read_csv("tmdb_5000_credits.csv")

    if 'movie_id' in credits.columns:
        credits.rename(columns={'movie_id':'id'}, inplace=True)

    credits = credits[['id','cast','crew']]

    df = movies.merge(credits, on='id')

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

df = load_data()

# -------------------------------
# Helper Functions
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

    if isinstance(text,str):
        return text.lower().replace(",","").replace(".","")

    return ""

# -------------------------------
# Feature Engineering
# -------------------------------
df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(lambda x: extract_names(x,limit=3))
df['director'] = df['crew'].apply(extract_director)
df['overview'] = df['overview'].apply(clean_text)

df['tags'] = (
    df['genres'] + " " +
    df['keywords'] + " " +
    df['cast'] + " " +
    df['director'] + " " +
    df['overview']
)

# -------------------------------
# Build ML Model
# -------------------------------
@st.cache_resource
def build_model(df):

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=8000,
        ngram_range=(1,2)
    )

    vectors = tfidf.fit_transform(df['tags'])

    similarity = cosine_similarity(vectors)

    return similarity

similarity = build_model(df)

# -------------------------------
# Similar Movie Recommendation
# -------------------------------
def recommend(movie_name):

    if movie_name not in df['title'].values:
        return pd.DataFrame()

    index = df[df['title']==movie_name].index[0]

    distances = similarity[index]

    movies = sorted(
        list(enumerate(distances)),
        key=lambda x:x[1],
        reverse=True
    )[1:6]

    return df.iloc[[i[0] for i in movies]]

# -------------------------------
# Mood Based Recommendation
# -------------------------------
mood_genres = {

    "Happy":["Comedy","Family","Animation"],
    "Sad":["Drama"],
    "Romantic":["Romance"],
    "Thriller":["Thriller","Mystery","Crime"],
    "Motivational":["Adventure","History","War"]

}

def recommend_by_mood(mood):

    genres = mood_genres[mood]

    mood_movies = df[
        df['genres'].str.contains('|'.join(genres), case=False)
    ]

    return mood_movies.sample(5)

# -------------------------------
# Sidebar UI
# -------------------------------
st.sidebar.title("🎥 Movie Recommender")

search = st.sidebar.text_input("🔍 Search movie")

filtered_movies = (
    df[df['title'].str.contains(search,case=False)]
    if search else df
)

selected_movie = st.sidebar.selectbox(
    "🎬 Choose a movie",
    filtered_movies['title'].values
)

# Mood Selection
mood = st.sidebar.selectbox(
    "😊 Choose your mood",
    ["Happy","Sad","Romantic","Thriller","Motivational"]
)

# -------------------------------
# Main UI
# -------------------------------
st.title("🎬 Movie Recommendation System")

st.markdown(
"Get recommendations based on **similar movies** or your **current mood** ✨"
)

# -------------------------------
# Similar Movies
# -------------------------------
if st.button("🔥 Show Similar Movies"):

    with st.spinner("Finding similar movies..."):

        recommendations = recommend(selected_movie)

    cols = st.columns(5)

    for i,movie in enumerate(recommendations.itertuples()):

        with cols[i]:

            st.markdown(f"### {movie.title}")
            st.caption(movie.overview[:120]+"...")

# -------------------------------
# Mood Movies
# -------------------------------
st.subheader(f"🎭 Movies for your {mood} mood")

if st.button("🎬 Show Mood Movies"):

    results = recommend_by_mood(mood)

    cols = st.columns(5)

    for i,movie in enumerate(results.itertuples()):

        with cols[i]:

            st.markdown(f"### {movie.title}")
            st.caption(movie.overview[:120]+"...")
