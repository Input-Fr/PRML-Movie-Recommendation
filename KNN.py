import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Charger les données
data = pd.read_csv('dataset/merged.csv')

# Créer une matrice utilisateur-item (userId, movieId, rating)
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Initialiser le modèle KNN
knn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn_model.fit(user_item_matrix)

# Fonction pour recommander des films pour un utilisateur donné
def recommend_movie_for_user(user_id):
    # Trouver les voisins les plus proches de l'utilisateur
    distances, indices = knn_model.kneighbors(user_item_matrix.loc[[user_id]], n_neighbors=5)

    # Obtenir les films préférés des voisins
    similar_users = user_item_matrix.iloc[indices[0]].index.tolist()
    recommended_movies = data[data['userId'].isin(similar_users)]
    recommended_movies = recommended_movies.groupby('movieId')['rating'].mean()
    recommended_movie = recommended_movies.idxmax()  # Trouver le film le mieux noté

    return recommended_movie

# Exemple d'utilisation
user_id = int(input("Enter user ID: "))
recommended_movie = recommend_movie_for_user(user_id)
movies = pd.read_csv('dataset/movies.csv')
title = movies['title'].where(movies['movieId'] == recommended_movie)
print("Top recommended movie for user", user_id, ":", title.dropna().values[0])

