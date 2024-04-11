import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

movieNgenre = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

merged = pd.merge(ratings, movieNgenre, on='movieId')
merged = merged.sort_values(by='userId')
# Reset the indexes
merged = merged.reset_index(drop=True)

# we normalize the rate so that the linear regression will perform better
scaler = MinMaxScaler()
merged['rating'] = scaler.fit_transform(merged[['rating']])

# we can drop title column as we won't need them we will use movieId instead
merged = merged.drop(columns=['title'])
# we also drop the timestamp column as we won't use it neither
merged = merged.drop(columns=['timestamp'])

# make a column for each genre
genres_dummies = merged['genres'].str.get_dummies(sep='|')

# concatenate encoded genres with the original DataFrame
merged = pd.concat([merged, genres_dummies], axis=1)

# drop 'genres' column and 'rating' column
X = merged.drop(columns=['genres', 'rating'])
y = merged["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(X)
X.to_csv("merged.csv")
y_pred = model.predict(X_test)

""" HOW TO INTERPRET/USE THE MODEL

Once the model is trained, it is able to predict how much a user is going to like
the movie through the prediction of the rating the user would give to this movie.

To know which movie to recommand to which user, we should get every movieID users
didn't rate yet, given this if the prediction says that the rating will be high
then we will be able to recommand the movie otherwise we won't recommand it.

"""

