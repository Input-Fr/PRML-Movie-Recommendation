import numpy as np
import pandas as pd
import random

from matplotlib import pyplot as plt

print("Collecting data...")

# Collect data
movies = pd.read_csv('archive/movies.csv')
ratings = pd.read_csv('archive/ratings.csv')

print(f"{len(movies)} movies and {len(ratings)} ratings")

print("Data treatment...")

# Simplify data
merged = pd.merge(ratings, movies, on='movieId')

merged = merged.drop(columns=['title'])
merged = merged.drop(columns=['timestamp'])

print(merged)

print("Listing genres...")

# Catch all existing movie genres
genres = []
for genres1 in merged["genres"]:
    for genre in genres1.split('|'):
        if not genre in genres and genre != '(no genres listed)':
            genres.append(genre)

print(f"Found {len(genres)} genres")

print("Converting data types...")

# Convert data type
merged["rating"] = merged["rating"].astype(float)
merged["userId"] = merged["userId"].astype(int)
merged["movieId"] = merged["movieId"].astype(int)

merged = merged.sort_values(by='movieId')

print("Assigning genres to movies...")

# Collect movies genre for all movieId
movie_genres = dict()
for i in range(len(merged)):
    cur = merged.il oc[i]
    types = cur["genres"].split('|')
    movieId = cur['movieId']

    typesId = []
    for t in types:
        if t in genres:
            typesId.append(genres.index(t))
    movie_genres[movieId] = typesId

print(f"Assigned {len(movie_genres)} genres")

print("Listing users...")

# Collect users and their ratings
users = dict()
for i in range(len(merged)):
    cur = merged.iloc[i]
    user = cur["userId"]
    rating = cur["movieId"], cur["rating"]

    if not user in users.keys():
        users[user] = []
    users[user].append(rating)

print(f"{len(users)} users")


class User:
    '''
        id= userId
        ratings= list of tuplet: (movieId,rating)
    '''

    def __init__(self, user_id=None, user_ratings=None):
        if user_ratings is None:
            user_ratings = []
        self.id = user_id
        self.ratings = user_ratings
        self.stats = self.genres_stats()

    def genres_stats(self):
        stats = []
        learning_rate = 0.1
        while learning_rate > 0:
            MovieId, rating = self.ratings[random.randint(0, len(self.ratings) - 1)]
            cur_genres = movie_genres[MovieId]
            for g in cur_genres:
                while len(stats) < g + 1:
                    stats.append(0)
                stats[g] += (rating - 3.5) / len(cur_genres) * learning_rate
            learning_rate -= 0.0001

        return stats

    def evaluate(self, movieId):
        res = 0
        for movie_genre in movie_genres[movieId]:
            if len(self.stats) > movie_genre:
                res += self.stats[movie_genre] / len(movie_genres[movieId])
        return res


'''
for i in range(len(users)):
    if len(users[i]) > 0:
        users[i] = User(i, users[i])
'''


def shuffle(list):
    list1 = list.copy()
    res = []
    while len(list1) > 0:
        i = random.randint(0, len(list1) - 1)
        ele = list1.pop(i)
        res.append(ele)
    return res

'''
def compare(user1, user2):
    ratings1, ratings2 = user1.ratings.copy(), user2.ratings.copy()
    n1, n2 = len(ratings1), len(ratings2)
    common1 = []
    common2 = []
    i, j = 0, 0
    while i < len(ratings1) and j < len(ratings2):
        if ratings1[i] > ratings2[j]:
            j += 1
        elif ratings1[i] < ratings2[j]:
            i += 1
        else:
            ele1 = ratings1.pop(i)
            ele2 = ratings2.pop(j)
            common1.append(ele1)
            common2.append(ele2)
'''



user_ratings = users[356]

print(user_ratings)

n = int(len(user_ratings) * 0.7)

shuffled = shuffle(user_ratings)

train, test = shuffled[:n], shuffled[n:]

# test = [(193583, 5.0)]

test_user = User(user_id=0, user_ratings=train)

print(test_user.stats)

avg_ratio = 0

ratio_list = []

x = []
y = []

for grade in test:
    moveId, score = grade
    res = test_user.evaluate(moveId)
    x.append(score)
    y.append(res)
    ratio = res / score
    print(f"{score}  --> predicted {res}  ratio {ratio}")
    ratio_list.append(ratio)
    avg_ratio += ratio

avg_ratio /= len(ratio_list)

variance = 0

for ratio in ratio_list:
    cur_dif = ratio - avg_ratio
    variance += cur_dif * cur_dif

variance /= len(ratio_list)

print(f"Average ratio: {avg_ratio} -- Variance: {variance}\nOverall performance: {variance / avg_ratio}")

x_values = np.linspace(0, 5, 100)

plt.scatter(x, y)

plt.show()