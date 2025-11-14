import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import requests
from bs4 import BeautifulSoup
import altair as alt 

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean/cleaned_data.csv")
    df["title"] = df["title"].astype(str).str.lower().str.strip()
    return df

df = load_data()

st.title(" Book Recommendation App")

tab1, tab2, tab3 = st.tabs(["Recommendations","Explore Books", "Dashboard"])

with tab1:
    # ------------------------------
    # Prepare Feature Matrix
    # ------------------------------
    @st.cache_resource
    def prepare_similarity(df):
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity

        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(df[["genre", "author"]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["genre", "author"]))

        features = pd.concat([
            df[["average_rating", "publication_year"]].reset_index(drop=True),
            encoded_df
        ], axis=1)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        return cosine_similarity(scaled_features)

    cosine_sim = prepare_similarity(df)

    # ------------------------------
    # Helper: Fetch Book Cover (Goodreads + OpenLibrary)
    # ------------------------------
    @st.cache_data
    def get_book_cover(link):
        """
        Fetch book cover image from Goodreads or OpenLibrary.
        Handles relative URLs (//covers.openlibrary.org...) and both site structures.
        """
        if not link:
            return "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

        headers = {"User-Agent": "Mozilla/5.0"}

        # --- Goodreads cover fetch ---
        if "goodreads.com" in link:
            try:
                response = requests.get(link, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")

                img_tag = soup.find("img", {"class": "ResponsiveImage"})
                if img_tag and img_tag.get("src"):
                    img_url = img_tag["src"]
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    return img_url
            except Exception:
                pass

        # --- OpenLibrary cover fetch ---
        if "openlibrary.org" in link:
            try:
                response = requests.get(link, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")

                # Primary pattern
                img_tag = soup.find("img", {"itemprop": "image"})
                if img_tag and img_tag.get("src"):
                    img_url = img_tag["src"]
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    return img_url

                # Fallback: extract numeric ID
                if "/b/id/" in link:
                    cover_id = link.split("/b/id/")[1].split("/")[0]
                    return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
            except Exception:
                pass

        # --- Default fallback image ---
        return "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

    # ------------------------------
    # Recommendation Function
    # ------------------------------
    def recommend_books(title, df, cosine_sim, n=5):
        from difflib import get_close_matches

        title_clean = title.strip().lower()
        matches = df[df["title"].str.lower().str.contains(title_clean, na=False)]

        if matches.empty:
            close = get_close_matches(title_clean, df["title"].str.lower(), n=1, cutoff=0.4)
            if close:
                matches = df[df["title"].str.lower() == close[0]]

        if matches.empty:
            st.warning(f"No books found for '{title}'. Try another keyword.")
            return None, None

        matched_idx = matches.index[0]
        matched_book = df.loc[matched_idx]

        sim_scores = list(enumerate(cosine_sim[matched_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i for i, _ in sim_scores[1:n+1]]

        recs = df.iloc[sim_indices][["title", "author", "genre", "average_rating", "link"]]
        return matched_book, recs

    # ------------------------------
    # Streamlit UI
    # ------------------------------
    st.title(" Book Recommendation System")
    st.markdown("### Find similar books based on author, genre, and rating!")

    book_input = st.text_input("Enter a book title (partial or full):", "")

    if book_input:
        matched_book, recommendations = recommend_books(book_input, df, cosine_sim, n=5)

        if matched_book is not None:
            st.markdown("## Book Details:")

            cover_url = get_book_cover(matched_book["link"])

            # Stylish card layout using HTML + Streamlit markdown
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                display: flex;
                align-items: flex-start;
                gap: 20px;
            ">
                <div style="flex-shrink: 0; text-align: center;">
                    <img src="{cover_url}" alt="Book Cover" style="width:150px; border-radius:10px;">
                </div>
                <div style="flex: 1;">
                    <h3 style="margin-bottom: 5px;">{matched_book['title'].title()}</h3>
                    <p style="margin: 0; font-size: 15px; color: #555;">
                        <em>by {matched_book['author']}</em><br>
                        <strong>Genre:</strong> {matched_book['genre']}<br>
                        ⭐ <strong>Rating:</strong> {matched_book['average_rating']}
                    </p>
                    <a href="{matched_book['link']}" target="_blank" 
                        style="display:inline-block; margin-top:10px; padding:6px 12px;
                        background-color:#2c6df2; color:white; text-decoration:none;
                        border-radius:6px; font-size:14px;">
                        🔗 View Book
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- Recommended Books Section ---
            st.markdown("### More to read")
            for i, row in recommendations.iterrows():
                cover_url = get_book_cover(row["link"])
                st.markdown(f"""
                <div style="display:flex; gap:10px; align-items:center; margin-bottom:10px;">
                    <img src="{cover_url}" alt="Book Cover" style="width:60px; border-radius:6px;">
                    <div>
                        <b>{row['title'].title()}</b><br>
                        <i>by {row['author']}</i><br>
                        <span style="color: gray;">Genre:</span> {row['genre']}<br>
                        ⭐ {row['average_rating']}
                        <a href="{row['link']}" target="_blank">🔗 View</a>
                    </div>
                </div>
                <hr style="margin: 6px 0;">
                """, unsafe_allow_html=True)
with tab2:
    # ------------------------------
    # Explore by Genre / Author
    # ------------------------------
    st.title("Explore Books")
    st.markdown("### Browse top-rated books by genre or author")

    # Dropdown filters
    genres = ["All"] + sorted(df["genre"].dropna().unique().tolist())

    selected_genre = st.selectbox("Select a genre:", genres)

    # Filter the DataFrame
    filtered_df = df.copy()
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df["genre"] == selected_genre]

    # Sort by rating (descending)
    filtered_df = filtered_df.sort_values(by="average_rating", ascending=False)

    # Limit the number of books displayed
    top_n = st.slider("Number of books to display:", 1, 10, 5)

    # ------------------------------
    # Display Results
    # ------------------------------
    if filtered_df.empty:
        st.warning("No books found for the selected filters.")
    else:
        st.markdown(f"### Showing Top {top_n} Books")
        for i, row in filtered_df.head(top_n).iterrows():
            cover_url = get_book_cover(row["link"])
            st.markdown(f"""
            <div style="
                display:flex;
                gap:15px;
                align-items:center;
                background-color:#f8f9fa;
                border:1px solid #e0e0e0;
                border-radius:12px;
                padding:12px;
                margin-bottom:10px;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);
            ">
                <img src="{cover_url}" alt="Book Cover" style="width:80px; border-radius:8px;">
                <div>
                    <b>{row['title'].title()}</b><br>
                    <i>by {row['author']}</i><br>
                    <span style="color:gray;">Genre:</span> {row['genre']}<br>
                    ⭐ {row['average_rating']}
                    <a href="{row['link']}" target="_blank" style="margin-left:6px;">🔗 View</a>
                </div>
            </div>
            """, unsafe_allow_html=True)


with tab3:
    st.subheader(" Dataset Overview & Insights")
    st.markdown("Explore statistics about genres, authors, ratings, and publication trends in your dataset.")

    # --- Summary Metrics ---
    total_books = len(df)
    total_authors = df['author'].nunique()
    avg_rating = df['average_rating'].mean()
    avg_year = int(df['publication_year'].mean())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Books", f"{total_books:,}")
    col2.metric("Unique Authors", f"{total_authors:,}")
    col3.metric("Avg. Rating", f"{avg_rating:.2f}")
    col4.metric("Avg. Publication Year", avg_year)

    st.markdown("---")

    # --- Row 1: Genre vs. Author Charts ---
    st.subheader("Genre & Author Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 Genres")
        genre_counts = (
            df['genre']
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .reset_index()
        )
        # Rename columns safely regardless of pandas naming
        genre_counts.columns = ["Genre", "Count"]
        genre_counts["Genre"] = genre_counts["Genre"].astype(str)
        genre_counts["Count"] = pd.to_numeric(genre_counts["Count"], errors="coerce").fillna(0)

        genre_chart = alt.Chart(genre_counts).mark_bar(color="#588157").encode(
            x=alt.X("Count:Q", title="Number of Books"),
            y=alt.Y("Genre:N", title="Genre", sort="-x"),
            tooltip=["Genre:N", "Count:Q"]
        ).properties(height=350)
        st.altair_chart(genre_chart, use_container_width=True)

    with col2:
        st.markdown("#### Top 10 Authors")
        author_counts = (
            df['author']
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .reset_index()
        )
        author_counts.columns = ["Author", "Count"]
        author_counts["Author"] = author_counts["Author"].astype(str)
        author_counts["Count"] = pd.to_numeric(author_counts["Count"], errors="coerce").fillna(0)

        author_chart = alt.Chart(author_counts).mark_bar(color="#bd3039").encode(
            x=alt.X("Count:Q", title="Number of Books"),
            y=alt.Y("Author:N", title="Author", sort="-x"),
            tooltip=["Author:N", "Count:Q"]
        ).properties(height=350)
        st.altair_chart(author_chart, use_container_width=True)


    # --- Row 2: Publication Century vs. Ratings ---
    st.subheader("Publication & Rating Trends")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Books Published by Century")

        df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')

        def get_century(year):
            if pd.isna(year):
                return "Unknown"
            elif year < 1900:
                return "19th Century"
            elif year < 2000:
                return "20th Century"
            else:
                return "21st Century"

        df['century'] = df['publication_year'].apply(get_century)

        century_data = (
            df['century'].value_counts()
            .reindex(["19th Century", "20th Century", "21st Century"])
            .reset_index()
        )
        century_data.columns = ["Century", "Count"]

        century_chart = alt.Chart(century_data).mark_bar(color="#255e9e").encode(
            x=alt.X("Century:N", sort=["19th Century", "20th Century", "21st Century", "Unknown"]),
            y=alt.Y("Count:Q", title="Number of Books"),
            tooltip=["Century", "Count"]
        ).properties(height=350)
        st.altair_chart(century_chart, use_container_width=True)

    with col4:
        st.markdown("####  Ratings Distribution")
        ratings_chart = alt.Chart(df).mark_bar(color="#f4b400").encode(
            x=alt.X("average_rating:Q", bin=alt.Bin(maxbins=20), title="Average Rating"),
            y=alt.Y("count()", title="Number of Books"),
            tooltip=["count()"]
        ).properties(height=350)
        st.altair_chart(ratings_chart, use_container_width=True)
