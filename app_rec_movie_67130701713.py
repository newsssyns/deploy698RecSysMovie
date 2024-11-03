
import streamlit as st
import pickle

# Load model and data
with open('67130701713recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Get personalized movie recommendations based on user ID")

# User input
user_id = st.number_input("Enter User ID:", min_value=1, value=1, step=1)

# Recommendation Logic
if st.button("Get Recommendations"):
    # Find movies that the user hasn't rated
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    
    # Predict ratings for unrated movies
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    
    # Get top 10 recommendations
    top_recommendations = sorted_predictions[:10]
    
    # Display top recommendations
    st.write(f"### Top 10 Movie Recommendations for User {user_id}")
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")
