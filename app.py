import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the MovieLens dataset
movies = pd.read_csv('movies1.csv')
ratings = pd.read_csv('ratings1.csv')

# Pivot the ratings DataFrame to create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Fit Nearest Neighbors model for collaborative filtering
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(user_item_matrix.T)

# TF-IDF Vectorizer for content-based recommendations
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])


# Function to get collaborative filtering movie recommendations
def get_cf_movie_recommendations(movie_id, num_recommendations=5):
    distances, indices = model_knn.kneighbors(user_item_matrix.T.iloc[movie_id].values.reshape(1, -1),
                                              n_neighbors=num_recommendations + 1)

    # Exclude the queried movie itself (closest neighbor)
    recommendations = [(movies.iloc[idx]['title'], distances[0][i]) for i, idx in enumerate(indices[0][1:])]

    return recommendations


# Function to get content-based movie recommendations
def get_content_based_recommendations(movie_title, num_recommendations=5):
    # Find the movie index
    movie_index = movies[movies['title'].str.contains(movie_title, case=False)].index[0]

    # Calculate cosine similarity between the queried movie and all others
    cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    # Get indices of movies with highest similarity scores
    content_based_recommendations = [(movies.iloc[idx]['title'], cosine_similarities[idx]) for idx in
                                     cosine_similarities.argsort()[:-num_recommendations - 1:-1]]

    return content_based_recommendations


# Streamlit App
st.image("https://c4.wallpaperflare.com/wallpaper/862/449/162/jack-reacher-star-wars-interstellar-movie-john-wick-wallpaper-preview.jpg", use_column_width=True)
st.title("Movie Recommendation App")

# Sidebar for user input
searched_movie_title = st.sidebar.text_input("Type the title of a movie:")

# Display collaborative filtering recommendations
if searched_movie_title:
    # Find the closest movie based on the searched title
    queried_movie_id = movies[movies['title'].str.contains(searched_movie_title, case=False)].index[0]

    # Get collaborative filtering recommendations for the queried movie
    cf_recommendations = get_cf_movie_recommendations(queried_movie_id)

    st.subheader(f"Collaborative Filtering Recommendations for '{searched_movie_title}':")
    for movie_title, similarity_score in cf_recommendations:
        st.write(f"{movie_title}: Similarity Score - {1 - similarity_score:.4f}")

# Display content-based recommendations
if searched_movie_title:
    # Get content-based recommendations for the queried movie title
    content_based_recommendations = get_content_based_recommendations(searched_movie_title)

    st.subheader(f"Content-Based Recommendations for '{searched_movie_title}':")
    for movie_title, similarity_score in content_based_recommendations:
        st.write(f"{movie_title}: Similarity Score - {similarity_score:.4f}")
