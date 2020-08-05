import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import math
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def train_model(in_movies, df, movielens_dir):

    # Download the actual data from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    # Use the ratings.csv file


    # Only extract the data the first time the script is run.
    if not movielens_dir.exists():
        with ZipFile(movielens_zipped_file, "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=keras_datasets_path)
            print("Done!")

    #rom=search_genre("Romance")
    lent = len(in_movies)

    df = df.append( pd.DataFrame({'userId':[9999]*lent,
                                 'movieId':in_movies,
                                 'rating':[10]*lent,
                                 'timestamp': [1431957425]*lent
                                  }))



    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df["rating"] = df["rating"].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    print(
        "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
            num_users, num_movies, min_rating, max_rating
        )
    )



    df = df.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    # Normalize the targets between 0 and 1. Makes it easy to train.
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    # Assuming training on 90% of the data and validating on 10%.
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )




    EMBEDDING_SIZE = 50


    class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_movies, embedding_size, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.num_users = num_users
            self.num_movies = num_movies
            self.embedding_size = embedding_size
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.movie_embedding = layers.Embedding(
                num_movies,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.movie_bias = layers.Embedding(num_movies, 1)

        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            movie_vector = self.movie_embedding(inputs[:, 1])
            movie_bias = self.movie_bias(inputs[:, 1])
            dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
            # Add all the components (including bias)
            x = dot_user_movie + user_bias + movie_bias
            # The sigmoid activation forces the rating to between 0 and 1
            return tf.nn.sigmoid(x)

    model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
    )


    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=2,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    return(model, movie2movie_encoded, user2user_encoded, movie_encoded2movie)


#
# # Let us get a user and see the top recommendations.
# user_id = df.userId.sample(1).iloc[0]
# user_id = 9999
# movies_watched_by_user = df[df.userId == user_id]
# movies_not_watched = movie_df[
#     ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
# ]["movieId"]
# movies_not_watched = list(
#     set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
# )
# movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
# user_encoder = user2user_encoded.get(user_id)
# user_movie_array = np.hstack(
#     ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
# )
# ratings = model.predict(user_movie_array).flatten()
# top_ratings_indices = ratings.argsort()[-15:][::-1]
# recommended_movie_ids = [
#     movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
# ]
#
# print("Showing recommendations for user: {}".format(user_id))
# print("====" * 9)
# print("Movies with high ratings from user")
# print("----" * 8)
# top_movies_user = (
#     movies_watched_by_user.sort_values(by="rating", ascending=False)
#     .head(5)
#     .movieId.values
# )
# movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
# for row in movie_df_rows.itertuples():
#     print(row.title, ":", row.genres)
#
# print("----" * 8)
# print("Top 10 movie recommendations")
# print("----" * 8)
# recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
# for row in recommended_movies.itertuples():
#     print(row.title, ":", row.genres)


def make_predictions(in_list: list, model, movie_df, df, movie2movie_encoded, user2user_encoded, movie_encoded2movie) -> list:

    movies_watched_by_user = df[df.movieId.isin(in_list)]
    print(movies_watched_by_user)
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    user_id=9999
    user_encoder = user2user_encoded.get(user_id)
    #print([[user_encoder]] * len(movies_not_watched))

    user_movie_array = np.hstack(
        ([[610]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-100:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices[:15]
    ]

    rest_recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices[16:]
    ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
        movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(10)
            .movieId.values
    )
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    rest_of_list = movie_df[movie_df["movieId"].isin(rest_recommended_movie_ids)]

    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)

    out_list = ' '.join([x + " <p>" for x in recommended_movies.title.tolist()[:10]])

    return(out_list, rest_of_list, recommended_movie_ids)
#
#
# movs1 = [77866, 4351, 474, 1610, 2987,42, 75]
# movs2 = [4005, 141, 981]
# movs3 = [1,3,4,5,7]
# movs4 = [17,25,28,46]
#
#
# rom=search_genre("Romance")
# lent = len(rom[:200])
# make_predictions(rom)
#
# import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def search_genre (ingenre: str, movie_df):
    movs = movie_df[movie_df.genres.apply(lambda x: x.split("|")[0]==ingenre and len(x.split("|"))==1)][['title','movieId','genres']]
    print(movs[['title','genres','movieId']])
    return(movs.movieId.tolist())

def find_mov (title: str):
    movs = movie_df[movie_df.title.str.contains(title)][['title','movieId', 'genres']]
    print(movs)

def find_genres_by_id (ids: str, movie_df):
    movs = movie_df[movie_df.movieId.isin(ids)]['genres'].tolist()
    genre_split = [' '.join(x.split('|')) for x in movs]

    out_list = []
    for x in genre_split:
        genre_split_out = out_list.extend(x.split())

    return(pd.Series(out_list))


def post_process(rec_ids: list, input_ids: list, cutoff: float, num_show_recs: int, pile, movie_df):
    rec_genres = find_genres_by_id(rec_ids, movie_df)
    input_genres = find_genres_by_id(input_ids, movie_df)

    input_genres_freq = input_genres.value_counts(normalize=True)
    output_genres_freq = rec_genres.value_counts(normalize=True)

    input_genres_above_cutoff = input_genres_freq[input_genres_freq >= cutoff]
    output_genres_above_cutoff = output_genres_freq[output_genres_freq >= cutoff]

    input_above_list = input_genres_above_cutoff.index.tolist()
    output_above_list = output_genres_above_cutoff.index.tolist()

    # how_many_to_add = math.floor(num_show_recs / 3)
    to_add_list = [x for x in input_above_list if x not in output_above_list]
    movs_to_add = pd.DataFrame()

    for mov in to_add_list:
        movs_to_add =  movs_to_add.append(pile[pile.genres.str.contains(mov)].sample(2))

    new_list = movie_df[movie_df.movieId.isin(rec_ids)].iloc[:(10-movs_to_add.shape[0]),:]
    new_list = new_list.append(movs_to_add)

    print(movs_to_add)
    print(new_list)

    return (new_list)
#post_process([13, 17, 20, 20, 20, 20,23, 25, 26], [11, 1, 3, 2, 3, 2, 2, 2, 2], .2, 3, movie_df, movie_df)

def main(in_list= [17,25,28,46]):

    movielens_data_file_url = (
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )
    movielens_zipped_file = keras.utils.get_file(
        "ml-latest-small.zip", movielens_data_file_url, extract=False
    )
    keras_datasets_path = Path(movielens_zipped_file).parents[0]
    movielens_dir = keras_datasets_path / "ml-latest-small"

    movie_df = pd.read_csv(movielens_dir / "movies.csv")

    ratings_file = movielens_dir / "ratings.csv"
    df = pd.read_csv(ratings_file)

    model, movie2movie_encoded, user2user_encoded, movie_encoded2movie = train_model(in_list,df, movielens_dir)
    preds = make_predictions(in_list, model, movie_df, df, movie2movie_encoded, user2user_encoded, movie_encoded2movie)

    post = post_process(preds[2], in_list, .2, 3, preds[1], movie_df)
    out_list = ' '.join([x + " <p>" for x in post.title.tolist()[:10]])
    fin = preds[0] + '<br>'  + "*****************" + "<br>" + out_list

    return(fin)

if __name__=='__main__':
    main()