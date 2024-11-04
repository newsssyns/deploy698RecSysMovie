import streamlit as st
import pickle
import pandas as pd


# Title of the Streamlit app
st.title("Movie Recommendation System")

# Load data from pickle file
with open('66130701713_recommendation_movie_svd.pkl', 'rb') as file:
    data = pickle.load(file)

# User ID input
user_id = st.number_input("Enter User ID:", min_value=1, value=1, step=1)

# Get recommendations for the user
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

# Sort predictions by estimated rating in descending order
sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)

# Get top 10 movie recommendations
top_recommendations = sorted_predictions[:10]

# Display recommendations
st.write(f"\nTop 10 movie recommendations for User {user_id}:")
for recommendation in top_recommendations:
    movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
    st.write(f"{movie_title} (Estimated Rating: {recommendation.est})")
