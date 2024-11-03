import streamlit as st
import pickle
import pandas as pd
from surprise import SVD

# Load data
with open('67130701713recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Set Streamlit title
st.title('Movie Recommender System')

# User input for user ID
user_id = st.number_input('Enter User ID', min_value=1, max_value=movie_ratings['userId'].max(), value=1)

# Get recommendations
if st.button('Get Recommendations'):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:10]

    # Display recommendations
    st.write(f"\nTop 10 movie recommendations for User {user_id}:")
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est})")
