
import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load the MovieLens dataset (ml-latest-small)
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Create a Surprise Reader object
reader = Reader(rating_scale=(1, 5))

# Load data into Surprise Dataset
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=.25)

# Train the SVD model
algo = SVD()
algo.fit(trainset)

# Function to get movie recommendations
def get_recommendations(user_id, top_n=10):
    # Get a list of movies the user has not rated
    user_ratings = ratings[ratings['userId'] == user_id]['movieId']
    unrated_movies = movies[~movies['movieId'].isin(user_ratings)]['movieId']

    # Predict ratings for unrated movies
    predictions = [algo.predict(user_id, movie_id).est for movie_id in unrated_movies]

    # Get top-N recommendations
    top_movie_ids = unrated_movies.iloc[pd.Series(predictions).sort_values(ascending=False).index[:top_n]]
    recommendations = movies[movies['movieId'].isin(top_movie_ids)]

    return recommendations

# Streamlit app
st.title("Movie Recommender System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=ratings['userId'].max(), value=1)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    st.write("Top 10 recommendations for User ID", user_id)
    st.dataframe(recommendations[['title', 'genres']])

