import streamlit as st
import pickle
from myfunction_67130701713 import get_movie_recommendations

# Load the pre-trained user similarity and movie ratings data
with open('67130701713recommendation_usersim.pkl', 'rb') as file:
    user_similarity_df, user_movie_ratings = pickle.load(file)

# Streamlit UI setup
st.title("Movie Recommendation System")
st.write("Get personalized movie recommendations based on user ID")

# User ID input
user_id = st.number_input("Enter User ID:", min_value=1, step=1, value=1)

# Display recommendations when button is clicked
if st.button("Get Recommendations"):
    recommendations = get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, 10)
    
    # Display the recommendations
    st.write(f"### Top 10 Movie Recommendations for User {user_id}")
    for movie_title in recommendations:
        st.write(f"• {movie_title}")
