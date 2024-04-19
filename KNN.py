import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load
data = pd.read_csv('dataset/merged.csv')

# create a user-item matrix (userId, movieId, rating)
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Initialize the KNN model
knn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn_model.fit(user_item_matrix)

# Function to recommend movies for a given user
def recommend_movie_for_user(user_id):
    # Find the nearest neighbors of the user
    distances, indices = knn_model.kneighbors(user_item_matrix.loc[[user_id]], n_neighbors=5)

    # Get the best movie ratings of the neighbors
    similar_users = user_item_matrix.iloc[indices[0]].index.tolist()
    recommended_movies = data[data['userId'].isin(similar_users)]
    recommended_movies = recommended_movies.groupby('movieId')['rating'].mean()
    recommended_movie = recommended_movies.idxmax()  # Trouver le film le mieux noté

    return recommended_movie

# Example of usage
user_id = int(input("Enter user ID: "))
recommended_movie = recommend_movie_for_user(user_id)
movies = pd.read_csv('dataset/movies.csv')
title = movies['title'].where(movies['movieId'] == recommended_movie)
print("Top recommended movie for user", user_id, ":", title.dropna().values[0])


