
!pip install streamlit
import streamlit as st
import pickle
from surprise import SVD

# Load the model and data
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit app title
st.title("Movie Recommendation System")

# User input for user ID
user_id = st.number_input("Enter User ID:", value=1, min_value=1, step=1)

# Function to get recommendations
def get_recommendations(user_id):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:10]
    return top_recommendations

# Get recommendations when user ID is entered
if user_id:
    recommendations = get_recommendations(user_id)

    # Display recommendations
    st.subheader(f"Top 10 movie recommendations for User {user_id}:")
    for recommendation in recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")
