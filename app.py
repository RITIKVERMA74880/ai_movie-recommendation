# =========================================================
# MOVIE RECOMMENDATION SYSTEM — CINEMATIC UI
# =========================================================

import numpy as np
import pandas as pd
import difflib
import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CineMatch — Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# GLOBAL CSS — CINEMATIC DARK THEME
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e8e8e8;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, #1a0a2e 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, #0d1a2e 0%, transparent 50%),
        #0a0a0f;
}

[data-testid="stHeader"] { background: transparent; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #3a1f6e; border-radius: 3px; }

/* ── Hero Title ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 8vw, 7rem);
    letter-spacing: 0.12em;
    line-height: 1;
    background: linear-gradient(135deg, #fff 30%, #a78bfa 65%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-tagline {
    font-size: 1rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #6b7280;
    margin-top: 0.5rem;
}
.hero-divider {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #7c3aed, transparent);
    margin: 1.5rem auto 0;
}

/* ── Search Section ── */
.search-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2.2rem 2.5rem 2rem;
    max-width: 760px;
    margin: 2rem auto 0;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}
.search-label {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #7c3aed;
    font-weight: 600;
    margin-bottom: 0.9rem;
}

/* Override Streamlit input */
[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(124,58,237,0.35) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.8rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    height: 52px !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
    outline: none !important;
    background: rgba(255,255,255,0.08) !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
    color: #4b5563 !important;
}
[data-testid="stTextInput"] > label { display: none !important; }
[data-testid="stTextInput"] > div { margin-bottom: 0 !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #a21caf 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    width: 100% !important;
    height: 52px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.5) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.5rem 0 1.5rem;
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(124,58,237,0.6), transparent);
}
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.15em;
    color: #e8e8e8;
    white-space: nowrap;
}

/* ── Match Badge ── */
.match-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #c4b5fd;
    margin: 0.5rem auto 1.5rem;
    text-align: center;
}

/* ── Movie Card ── */
.movie-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 1.5rem;
    cursor: pointer;
}
.movie-card:hover {
    transform: translateY(-6px);
    border-color: rgba(124,58,237,0.5);
    box-shadow: 0 16px 40px rgba(124,58,237,0.2);
}
.movie-card img {
    width: 100%;
    display: block;
    border-radius: 14px 14px 0 0;
    aspect-ratio: 2/3;
    object-fit: cover;
}
.card-body {
    padding: 0.75rem 0.75rem 1rem;
}
.card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #f3f4f6;
    text-align: center;
    line-height: 1.3;
    white-space: normal;
    word-wrap: break-word;
    margin-bottom: 0.5rem;
}
.card-link {
    display: block;
    text-align: center;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    text-decoration: none;
    font-weight: 600;
    opacity: 0.8;
    transition: opacity 0.2s;
}
.card-link:hover { opacity: 1; }

/* ── Spinner override ── */
[data-testid="stSpinner"] > div {
    border-top-color: #7c3aed !important;
}

/* ── Hide default streamlit footer / menu ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Progress / info text ── */
[data-testid="stMarkdownContainer"] p {
    font-family: 'DM Sans', sans-serif;
}

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 0.35rem !important; }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero-wrap">
    <p class="hero-logo">CineMatch</p>
    <p class="hero-tagline">AI-Powered Movie Recommendations</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TMDB API KEY
# =========================================================

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"


# =========================================================
# FETCH POSTER FUNCTION
# =========================================================

def fetch_poster(movie_title):
    try:
        movie_title = movie_title.strip()
        search_url = (
            "https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}"
            f"&query={requests.utils.quote(movie_title)}"
        )
        response = requests.get(search_url, timeout=5)
        data = response.json()

        if 'results' not in data or len(data['results']) == 0:
            return "https://via.placeholder.com/300x450/1a1a2e/7c3aed?text=No+Poster"

        for movie in data['results']:
            if movie.get('poster_path'):
                return "https://image.tmdb.org/t/p/w500" + movie['poster_path']

        return "https://via.placeholder.com/300x450/1a1a2e/7c3aed?text=No+Poster"

    except Exception:
        return "https://via.placeholder.com/300x450/1a1a2e/7c3aed?text=Error"


# =========================================================
# LOAD DATASETS (cached for performance)
# =========================================================

@st.cache_data(show_spinner=False)
def load_and_prepare_data():

    movies1 = pd.read_csv(
        r'movies.csv'
    )
    movies2 = pd.read_csv(
        r'bollywood_movies.csv'
    )
    movies3 = pd.read_csv(
        r'tmdb_5000_movies.csv'
    )

    # Fix Bollywood columns
    movies2.rename(columns={
        'movie_name': 'title',
        'genre': 'genres',
        'lead_actor': 'cast'
    }, inplace=True)

    required_columns = ['title', 'genres', 'keywords', 'tagline', 'cast', 'director']

    for dataset in [movies1, movies2, movies3]:
        for col in required_columns:
            if col not in dataset.columns:
                dataset[col] = ''

    movies1 = movies1[required_columns]
    movies2 = movies2[required_columns]
    movies3 = movies3[required_columns]

    for dataset in [movies1, movies2, movies3]:
        dataset['title'] = dataset['title'].astype(str).str.strip()
        dataset.dropna(subset=['title'], inplace=True)

    movies_data = pd.concat([movies1, movies2, movies3], ignore_index=True)

    movies_data = movies_data[
        ~movies_data['title'].str.contains(r'^Movie_\d+', regex=True, na=False)
    ]

    movies_data.reset_index(drop=True, inplace=True)
    movies_data.reset_index(inplace=True)

    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = (
        movies_data['genres'] + ' ' +
        movies_data['keywords'] + ' ' +
        movies_data['tagline'] + ' ' +
        movies_data['cast'] + ' ' +
        movies_data['director']
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    list_of_all_titles = movies_data['title'].tolist()
    lowercase_titles = [t.lower() for t in list_of_all_titles]

    return movies_data, similarity, list_of_all_titles, lowercase_titles


# =========================================================
# LOAD WITH SPINNER
# =========================================================

with st.spinner("⚡ Initialising AI engine…"):
    movies_data, similarity, list_of_all_titles, lowercase_titles = load_and_prepare_data()

st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:0.8rem;"
    " letter-spacing:0.1em; margin-bottom:1rem;'>"
    f"✦ {len(list_of_all_titles):,} movies indexed &nbsp;·&nbsp; AI ready</p>",
    unsafe_allow_html=True
)


# =========================================================
# SEARCH PANEL — label + input + button all inside one box
# =========================================================

# Outer centering: blank | content | blank
_, center_col, _ = st.columns([0.5, 3, 0.5])

with center_col:

    # Styled card container (label only — widgets go below inside same column)
    st.markdown("""
    <div class="search-container">
        <p class="search-label">🎬 &nbsp;What movie do you love?</p>
    </div>
    """, unsafe_allow_html=True)

    # Negative margin to visually "pull" the widgets up into the card
    st.markdown(
        "<div style='margin-top:-1.6rem;'>",
        unsafe_allow_html=True
    )

    input_col, btn_col = st.columns([4, 1.3])

    with input_col:
        movie_name = st.text_input(
            "movie",
            placeholder="e.g. Inception, Kabhi Khushi Kabhie Gham…",
            label_visibility="hidden"
        )

    with btn_col:
        st.markdown(
            "<div style='padding-top:0.15rem;'>",
            unsafe_allow_html=True
        )
        recommend_clicked = st.button("✦ Recommend", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# RECOMMENDATION LOGIC
# =========================================================

if recommend_clicked:

    if movie_name.strip() == "":
        st.warning("⚠️ Please enter a movie name to get recommendations.")

    else:

        find_close_match = difflib.get_close_matches(
            movie_name.lower(),
            lowercase_titles,
            n=1,
            cutoff=0.4
        )

        if len(find_close_match) == 0:
            st.error("🎭 Movie not found. Try a different title or check the spelling.")

        else:
            close_match = list_of_all_titles[
                lowercase_titles.index(find_close_match[0])
            ]

            # ── Match badge ──
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span class='match-badge'>✦ Matched: {close_match}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            index_of_the_movie = movies_data[
                movies_data.title == close_match
            ]['index'].values[0]

            similarity_score = list(enumerate(similarity[index_of_the_movie]))

            sorted_similar_movies = sorted(
                similarity_score,
                key=lambda x: x[1],
                reverse=True
            )

            # ── Section title ──
            st.markdown("""
            <div class="section-header">
                <div class="section-header-line"></div>
                <span class="section-title">Recommended For You</span>
                <div class="section-header-line" style="background: linear-gradient(90deg, transparent, rgba(124,58,237,0.6));"></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Collect top 20 recommendations ──
            printed_movies = set()
            recommendations = []

            for movie in sorted_similar_movies:
                index = movie[0]
                title_from_index = movies_data[
                    movies_data.index == index
                ]['title'].values[0]

                if (
                    title_from_index not in printed_movies
                    and title_from_index.lower() != close_match.lower()
                ):
                    printed_movies.add(title_from_index)
                    recommendations.append(title_from_index)

                if len(recommendations) >= 20:
                    break

            # ── Fetch posters with progress bar ──
            progress = st.progress(0, text="Fetching posters…")
            posters = []

            for i, title in enumerate(recommendations):
                posters.append(fetch_poster(title))
                progress.progress(
                    (i + 1) / len(recommendations),
                    text=f"Loading poster {i+1} of {len(recommendations)}…"
                )

            progress.empty()

            # ── Render cards in a 5-column grid ──
            cols = st.columns(5)

            for i, (title, poster) in enumerate(zip(recommendations, posters)):

                search_link = (
                    "https://www.google.com/search?q="
                    + title.replace(" ", "%20")
                    + "+movie"
                )

                with cols[i % 5]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <a href="{search_link}" target="_blank" style="text-decoration:none;">
                            <img src="{poster}" alt="{title}" loading="lazy"/>
                            <div class="card-body">
                                <div class="card-title">{title}</div>
                                <span class="card-link">Search ↗</span>
                            </div>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Footer note ──
            st.markdown(
                "<p style='text-align:center; color:#374151; font-size:0.75rem;"
                " margin-top:2rem; letter-spacing:0.08em;'>"
                "Recommendations powered by TF-IDF cosine similarity &nbsp;·&nbsp; Posters via TMDB</p>",
                unsafe_allow_html=True
            )
