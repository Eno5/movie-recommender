from zipfile import ZipFile as _zip # TODO won't need once databases are made
from io import BytesIO as _BIO # TODO won't need once databases are made
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import sys

data = _zip('raw data.zip')

def loadData(filename, cols=None): # TODO won't need once databases are made
    return pd.read_csv(_BIO(data.read(f'ml-25m/{filename}.csv')), usecols=cols)

def norm(series):
    return (series - series.min()) / (series.max() - series.min())

def filterUsers(ratings): # TODO won't need once database is made
    """Excludes users with too many reviews or are too biased.
    
    :: Params ::
    ratings - user ratings DataFrame
    """
    user_ratings = ratings.groupby('userId')['rating'].agg(['mean','count'])
    score_avg = user_ratings['mean'].mean()
    score_std = user_ratings['mean'].std()
    users_too_biased = user_ratings['mean'][lambda x: (x>=score_avg+score_std*2)
                                            | (x<=score_avg-score_std*2)].dropna()

    reviews_avg = user_ratings['count'].mean()
    reviews_std = user_ratings['count'].std()
    users_too_many_reviews = user_ratings['count'][lambda x: x>reviews_avg+reviews_std*2].dropna()

    return ratings[~ratings['userId'].isin(users_too_biased.index)
                    & ~ratings['userId'].isin(users_too_many_reviews.index)].reset_index()

def filterMovies(ratings, n=40): # TODO won't need once ratings database is made
    """Excludes movies with too few reviews.
    
    :: Params ::
    ratings - user ratings DataFrame
    n - least number of reviews to include
    """
    movie_count = ratings.groupby('movieId')['userId'].count()
    movies_too_few_reviews = movie_count[lambda x: x<n].dropna()

    return ratings[~ratings['movieId'].isin(movies_too_few_reviews.index)].reset_index()

def findSimilarMovies(selected_movie, ratings): # TODO ratings to be database
    """Returns array of other movies liked by similar users (at least 3/5 stars).
    
    :: Params ::
    selected_movie - movieId of the base movie
    ratings - user ratings DataFrame
    """

    # similar_users = ratings[(ratings['movieId']==selected_movie)
    #                         & (ratings['rating']>=3)]['userId'].unique()
    similar_users = ratings.query(f'movieId=={selected_movie} & rating>=3')['userId'].unique()
    print(f'Number of similar users: {len(similar_users)}')
    
    return ratings[ratings['userId'].isin(similar_users)]

def getSimilarity(selected_movie, df):
    """Finds cosine similarity between selected_movie and similar movies.
    
    :: Params ::
    selected_movie - movieId of the base movie
    df - DataFrame to use, has data on selected_movie and similar movies
            index is movieId
    """
    try:
        df.loc[selected_movie]
    except KeyError: # selected_movie not in DataFrame
        # find most prominent features of similar movies and normalize
        most_similar = df.sum()
        df.loc[selected_movie] = norm(most_similar)

    return pd.DataFrame(cosine_similarity(df, df),
                        index=df.index,
                        columns=df.index)[selected_movie]

def findGenreSimilarity(selected_movie, similar_movies, movies): # TODO movies to be database
    """Finds genre similarity between selected_movie and similar_movies.
    
    :: Params ::
    selected_movie - movieId of the base movie
    similar_movies - iterable with movies liked by similar users
    movies - movies DataFrame with genres, movieId's, and titles
    """
    df = movies[movies['movieId'].isin(similar_movies)].set_index('movieId').drop(columns='title')
    return getSimilarity(selected_movie, df)

def findTagSimilarity(selected_movie, similar_movies, tags): # TODO tags to be database
    """Finds tag similarity between selected_movie and similar_movies.
    
    :: Params ::
    selected_movie - movieId of the base movie
    similar_movies - iterable with movies liked by similar users
    tags - tags DataFrame with tag relevance per movieId
    """
    df = tags[tags['movieId'].isin(similar_movies)].pivot(index='movieId',
                                                            columns='tagId',
                                                            values='relevance')

    return getSimilarity(selected_movie, df)

def preProcess(selected_movie, ratings, movies, tags): # TODO ratings, movies, tags to be databases
    print('Finding similar movies (Will be a database query in future)')
    similar_movies_data = findSimilarMovies(selected_movie, ratings)
    similar_movies = similar_movies_data['movieId'].unique()
    print(f'Number of similar movies: {len(similar_movies)}')
    similar_movie_ratings = similar_movies_data.groupby('movieId')['rating'].agg(['count','mean'])
    del similar_movies_data

    start_time = time() # TODO remove from production
    print('Finding genre similarities')
    genre_sim = findGenreSimilarity(selected_movie, similar_movies, movies)
    print('Finding tag similarities')
    tag_sim = findTagSimilarity(selected_movie, similar_movies, tags)

    print('Merging datasets')
    movie_data = pd.merge(similar_movie_ratings, genre_sim, left_index=True, right_index=True)
    movie_data = pd.merge(movie_data, tag_sim, left_index=True, right_index=True)
    movie_data.columns = ['review_count','avg_rating','genre_sim','tag_sim']
    try:
        movie_data.drop(selected_movie, inplace=True)
    except KeyError: # selected_movie already dropped
        pass

    desc = movie_data.describe()

    hist = {'avg_rating': desc.loc['mean','avg_rating'],
            'genre_sim_thresh': desc.loc['75%','genre_sim'],
            'tag_sim_thresh': desc.loc['75%','tag_sim'],
            'avg_count': desc.loc['50%','review_count']}

    return movie_data, hist, start_time

def recommendations(movie_data, hist):
    """Generates top picks and interesting finds from similar movies and stats."""
    genre_sim = hist['genre_sim_thresh']
    tag_sim = hist['tag_sim_thresh']
    avg_count = hist['avg_count']
    avg_rating = hist['avg_rating']
    # similar = movie_data.query(f'genre_sim >= {genre_sim} & tag_sim >= {tag_sim}')
    similar = f'genre_sim >= {genre_sim} & tag_sim >= {tag_sim}'
    
    top_10_count = movie_data.quantile(0.9).loc['review_count']
    # top_picks_query = f'review_count >= {top_10_count} & avg_rating >= {avg_rating} & {similar}'
    # interesting_query = f'review_count >= {avg_count} & avg_rating >= {avg_rating+0.5} & {similar}'
    top_picks_query = f'review_count >= {top_10_count} & avg_rating >= {avg_rating} & genre_sim >= {genre_sim} & tag_sim >= {tag_sim}'
    interesting_query = f'review_count >= {avg_count} & avg_rating >= {avg_rating+0.5} & genre_sim >= {genre_sim} & tag_sim >= {tag_sim}'

    top_picks = movie_data.query(top_picks_query).sort_values(by=['tag_sim',
                                                                'review_count',
                                                                'avg_rating'], 
                                                            ascending=False).head(10)

    interesting_finds = movie_data.query(interesting_query).sort_values(by=['genre_sim',
                                                                        'tag_sim',
                                                                        'review_count',
                                                                        'avg_rating'], 
                                                                    ascending=False)
    interesting_finds = interesting_finds[~interesting_finds.index.isin(top_picks.index)].head(10)

    return top_picks, interesting_finds

if __name__=="__main__":
    selected_movie = int(sys.argv[1])

    print('Loading datasets (Will be databases in future)')
    ratings = loadData('ratings', cols=['userId','movieId','rating'])
    movies = pd.read_csv('movies_processed.csv') # movieId, title, genres (boolean)
    tags = loadData('genome-scores')
    # genome_tags = loadData('genome-tags') # tagId, tag
    # links = loadData('links', cols=['movieId, imdbId']) # http://www.imdb.com/title/tt{imdbId}, leading 0's needed if len(imdbId) < 7

    print('Filtering ratings dataset (Will be in database in future)')
    ratings = filterMovies(filterUsers(ratings))

    print('Gathering data on similar movies')
    movie_data, hist, start_time = preProcess(selected_movie, ratings, movies, tags)

    top_picks, interesting = recommendations(movie_data, hist)

    def get_titles(df, movies=movies):
        """Returns titles of recommendations."""
        return pd.merge(df, movies[['movieId','title']], left_index=True, right_on='movieId')['title']
    
    top_picks = get_titles(top_picks)
    interesting = get_titles(interesting)

    print('Took {:2f}s to complete'.format(time()-start_time))

    print('Top Picks:')
    print(top_picks.values)
    print('Also Check Out:')
    print(interesting.values)