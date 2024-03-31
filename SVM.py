from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

user_df = pd.read_csv("dataset/merged.csv") # userId  movieId  rating  genres
movies_df = pd.read_csv("dataset/movies.csv") # movieId  title   genres

user_to_train = 1

#hot ones for user
user_df_hot_one = user_df['genres'].str.get_dummies('|')
user_df = pd.concat([user_df, user_df_hot_one], axis=1)
user_df.drop(columns=['genres'], inplace=True)

#hot ones for movies
movies_df_hot_one = movies_df['genres'].str.get_dummies('|')
movies_df = pd.concat([movies_df, movies_df_hot_one], axis=1)
movies_df.drop(columns=['genres'], inplace=True)

#prepare value for training
tested_df = user_df[user_df['userId'] == user_to_train]
columns_to_adjust = tested_df.columns[tested_df.columns.get_loc('rating')+1:]
# Ajouter ('rating' - 2) à chaque élément de ces colonnes
for column in columns_to_adjust:
    tested_df.loc[:,column] += tested_df['rating'] - 2


#Training
X = tested_df.drop(columns=["rating"])
y = tested_df["rating"]
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialiser le modèle SVM
model = svm.SVC(kernel='linear')
#train
model.fit(X_train, y_train)
# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Accuracy check?
accuracy = accuracy_score(y_test, predictions)
print("Accuracy :", accuracy)