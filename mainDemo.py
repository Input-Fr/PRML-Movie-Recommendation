import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

###################### PRE-PROCESSING ######################

movieNgenre = pd.read_csv('dataset/movies.csv')
ratings = pd.read_csv('dataset/ratings.csv')

merged = pd.merge(ratings, movieNgenre, on='movieId')
merged = merged.sort_values(by='userId')
# Reset the indexes
merged = merged.reset_index(drop=True)

# we normalize the rate so that the linear regression will perform better
scaler = MinMaxScaler()
merged['rating'] = scaler.fit_transform(merged[['rating']])


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


# Remove column names and index
merged.columns = [None] * len(merged.columns)
merged.index = [None] * len(merged.index)

# Convert DataFrame to NumPy array
merged = merged.values


def setupLstOfUser(merged):
    lst = [User(merged[0][0])]
    for row in merged:
        if lst[-1].id != row[0]:
            lst.append(User(row[0], [(row[1], row[2])]))
        lst[-1].ratings.append((row[1], row[2]))
    return lst


userList = setupLstOfUser(merged)


def similarityDegree(user1, user2):
    similarityNote = 0
    r1 = user1.ratings.copy()
    r2 = user2.ratings.copy()
    for elt in r1:
        if elt in r2:  # they gave same note to same film, they are very similar
            r1.remove(elt)
            similarityNote += 1
        else:
            # first we check if they have both rated the movie
            for e in r2:
                if elt[0] == e[0]:  # if they have both rated the movie
                    similarityNote += min(elt[1],
                                          e[1])  # we add to the similarity note the lowest rate between the two users
                    break
    return similarityNote


def getHighestRatedMovie(ratings):
    max_rating = float('-inf')  # Initialize max_rating to negative infinity
    highest_rated_movie_id = None

    for movie_id, rating in ratings:
        if rating > max_rating:
            max_rating = rating
            highest_rated_movie_id = movie_id

    return highest_rated_movie_id


def recommend(user):
    """
    parameter : user instance of User class containing every of its preferences

    return : the list of ids of the movies that should be interesting for the user
    """
    degrees = []
    for otherUser in userList:
        if otherUser.id != user.id:
            degrees.append(similarityDegree(user, otherUser))
        else:
            degrees.append(-1)  # don't recommand urself

    sibIdx = np.argmax(np.array(degrees))
    sib = userList[sibIdx]  # get the one that has most similarity with

    recommendation = []
    # look for a film that the other one did not have seen yet
    r1 = sib.ratings
    r2 = user.ratings
    for movie, _ in r1:
        for movieU, _ in r2:
            if movie != movieU and not userAlreadySee(user, movie):
                recommendation.append(movie)
                break

    return recommendation[1:]


def isEveryElementInList(list1, list2):
    for item in list1:
        if item not in list2:
            return False
    return True


def userAlreadySee(user, movie):
    for rates in user.ratings:
        if movie == rates[0]:
            return True
    return False


def preciseRecommendation(user):
    """
    Based on the recommendation list, we will pick the movie that the user
    prefer based on its favourite movie genre
    """
    id = getHighestRatedMovie(user.ratings)  # get one of its favourite movie
    preciseReco = []
    # get the genre
    genres = np.array(movieNgenre[movieNgenre['movieId'] == id]["genres"])[0].split("|")
    # pick only movie of the same genre
    reco = recommend(user)

    for movie in reco:
        genresReco = np.array(movieNgenre[movieNgenre['movieId'] == movie]["genres"])[0].split("|")
        if isEveryElementInList(genres, genresReco)and not userAlreadySee(user, movie):
            preciseReco.append(movie)

    return preciseReco


############################################################


######################## Perceptron ########################

def perceptron(recommendations):
    # Collect data
    movies = pd.read_csv('dataset/movies.csv')
    ratings = pd.read_csv('dataset/ratings.csv')


    # Simplify data
    merged = pd.merge(ratings, movies, on='movieId')

    merged = merged.drop(columns=['title'])
    merged = merged.drop(columns=['timestamp'])


    # Catch all existing movie genres
    genres = []
    for genres1 in merged["genres"]:
        for genre in genres1.split('|'):
            if not genre in genres and genre != '(no genres listed)':
                genres.append(genre)



    # Convert data type
    merged["rating"] = merged["rating"].astype(float)
    merged["userId"] = merged["userId"].astype(int)
    merged["movieId"] = merged["movieId"].astype(int)

    merged = merged.sort_values(by='movieId')


    # Collect movies genre for all movieId
    movie_genres = dict()
    for i in range(len(merged)):
        cur = merged.iloc[i]
        types = cur["genres"].split('|')
        movieId = cur['movieId']

        typesId = []
        for t in types:
            if t in genres:
                typesId.append(genres.index(t))
        movie_genres[movieId] = typesId



    # Collect users and their ratings
    users = dict()
    for i in range(len(merged)):
        cur = merged.iloc[i]
        user = cur["userId"]
        rating = cur["movieId"], cur["rating"]

        if not user in users.keys():
            users[user] = []
        users[user].append(rating)

    class UserPerceptron:
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
                if len(self.stats) > movie_genre: # from perceptron to percep rond
                    res += self.stats[movie_genre] / len(movie_genres[movieId])
            return res

    test_perceptron = UserPerceptron(0,test_ratings)

    scores = []

    for movie in recommendations:
        score = test_perceptron.evaluate(movie)
        scores.append(score)


    finalReco = []
    for i in range(10):
        idxMax = np.argmax(np.array(scores))
        finalReco.append(recommendations[idxMax])
        scores.pop(idxMax)
        recommendations.pop(idxMax)

    return finalReco



############################################################


########################### SVM ############################

from sklearn import svm

def SVM(user, recommendations):
    '''user_df = pd.read_csv("merged.csv")  # userId  movieId  rating  genres
    movies_df = pd.read_csv("movies.csv")  # movieId  title   genres'''

    user_df = pd.DataFrame(columns=['userId', 'movieId', '(no genres listed)', 'Action', 'Adventure',
       'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
       'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Rating'])

    genre_list = ['(no genres listed)', 'Action', 'Adventure',
       'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
       'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    for rate in user.ratings:
        movie, score = rate # réel
        data = movieNgenre[movieNgenre['movieId'] == movie]
        genres = np.array(data['genres'])[0].split("|")
        row = {'userId': user.id, 'movieId': movie}
        n = int(score*10)

        row['Rating'] = n

        # print(row)
        for genre in genre_list:
            if genre in genres:
                row[genre] = 1
            else:
                row[genre] = 0

        user_df.loc[len(user_df)] = pd.Series(row)

    # Training
    X = user_df.drop(columns=["Rating"])
    y = user_df["Rating"]

    # Initialiser le modèle SVM
    model = svm.SVC(kernel='linear')
    # train



    model.fit(X, np.array(y))

    # Faire des prédictions sur l'ensemble de test

    rec_df = pd.DataFrame(columns=['userId', 'movieId', '(no genres listed)', 'Action', 'Adventure',
                                    'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                    'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
                                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])


    for reco in recommendations:
        movie = reco
        data = movieNgenre[movieNgenre['movieId'] == movie]
        genres = np.array(data['genres'])[0].split("|")
        row = {'userId': user.id, 'movieId': movie}

        for genre in genre_list:
            if genre in genres:
                row[genre] = 1
            else:
                row[genre] = 0

        rec_df.loc[len(rec_df)] = pd.Series(row)


    predictions = model.predict(rec_df)


    data = list(zip(recommendations, predictions))


    # Sort the list of tuples based on the scores
    sorted_data = sorted(data, key=lambda x: x[1])

    sorted_names = [item[0] for item in sorted_data]

    return sorted_names[len(sorted_names)-11:]

############################################################



##################### Using the models #####################

test_ratings = [(5349, 4.0), (44020, 5.0), (86332, 4.5), (260, 5.0), (8636, 4.0)] #, (65261, 1.0), (66097, 1.5), (122904, 5.0), (95510, 4.5), (52722, 4.0), (110553, 4.0)]

test_user = User(0,test_ratings)

# First preprocess the datas
recoList = recommend(test_user)

firstC = input("How do you want to set user's preferences ?\n\033[94m1\033[0m Rating random movies\n\033[94m2\033[0m Custom choosen movies (in code)\n")
while firstC != '1' and firstC != '2' and firstC != '3':
    print("Invalid choice")
    firstC = input("Which model do you want to use ?\n\033[94m1\033[0m Support Vector Machine (Recommended)\n\033[94m2\033[0m Perceptron\n")


if firstC == "1":
    test_ratings = []
    test_user = User(0, test_ratings)
    for i in range(10):
        # select randomly a movie among movie list
        row = movieNgenre.sample(n=1)
        rateGiven = input(f"Give a rate out of 50 of the following movie \"{np.array(row['title'])[0]}\"\n")
        rate = float(rateGiven)/10
        movieId = np.array(row['movieId'])[0]
        # add movie in list
        test_user.ratings.append((movieId, rate))


    recoList = recommend(test_user)

# Then use a model
choice = input("Which model do you want to use ?\n\033[94m1\033[0m Support Vector Machine (Recommended)\n\033[94m2\033[0m Perceptron\n")

while choice != '1' and choice != '2' and choice != '3':
    print("Invalid choice")
    choice = input("Which model do you want to use ?\n\033[94m1\033[0m Support Vector Machine (Recommended)\n\033[94m2\033[0m Perceptron\n")

movieIdRecommended = []
if choice == "1":
    print("SVM choosen")
    movieIdRecommended = SVM(test_user, recoList)
elif choice == "2":
    print("Perceptron choosen")
    movieIdRecommended = perceptron(recoList)

for _,row in movieNgenre.iterrows():
    if int(row["movieId"]) in movieIdRecommended:
        print(row["title"])


############################################################




