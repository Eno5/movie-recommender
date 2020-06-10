import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys
import sqlite3

conn = sqlite3.connect('movie_ratings.db')

def norm(series):
    return (series - series.min()) / (series.max() - series.min())

def findSimilarMovies(selected_movie, c=conn):
    """Creates temp table of other movies rated by similar users (at least 3/5 stars).
        Also returns DataFrame with avg ratings and number of reviews for other movies.
    
    :: Params ::
    selected_movie - movieId of the base movie
    c - connection to database
    """
    SQL = """
        WITH users AS (
            SELECT DISTINCT userId FROM ratings
            WHERE movieId=? AND rating >= 3
        )
        SELECT movieId, rating FROM ratings
        WHERE userId IN (SELECT userId FROM users LIMIT 10000)
    """

    movie_ratings = pd.read_sql(SQL, c, params=(selected_movie,)).groupby('movieId')['rating'].agg(['count','mean'])
    return movie_ratings

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

def findGenreSimilarity(selected_movie, movies, c=conn):
    """Finds genre similarity between selected_movie and similar movies.
    
    :: Params ::
    selected_movie - movieId of the base movie
    movies - similar movies DataFrame
    c - connection to database
    """
    df = pd.read_sql('SELECT * FROM movies', c)
    df = df[df['movieId'].isin(movies.index)].set_index('movieId').drop(columns='title')

    return getSimilarity(selected_movie, df)

def findTagSimilarity(selected_movie, movies, c=conn):
    """Finds tag similarity between selected_movie and similar movies.
    
    :: Params ::
    selected_movie - movieId of the base movie
    movies - similar movies DataFrame
    c - connection to database
    """
    df = pd.read_sql('SELECT * FROM tags', c)    
    df = df[df['movieId'].isin(movies.index)].pivot(index='movieId',
                                                    columns='compId',
                                                    values='score')

    return getSimilarity(selected_movie, df)

def preProcess(selected_movie):
    print('Finding similar movies')
    movie_ratings = findSimilarMovies(selected_movie)
    print('Finding genre similarities')
    genre_sim = findGenreSimilarity(selected_movie, movie_ratings)
    print('Finding tag similarities')
    tag_sim = findTagSimilarity(selected_movie, movie_ratings)

    print('Merging datasets')
    movie_data = pd.merge(movie_ratings, genre_sim, left_index=True, right_index=True)
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

    return movie_data, hist

def recommendations(movie_data, hist):
    """Generates top picks and interesting finds from similar movies and stats."""
    genre_sim = hist['genre_sim_thresh']
    tag_sim = hist['tag_sim_thresh']
    avg_count = hist['avg_count']
    avg_rating = hist['avg_rating']
    similar = f'genre_sim >= {genre_sim} & tag_sim >= {tag_sim}'
    
    top_10_count = movie_data.quantile(0.9).loc['review_count']
    top_picks_query = f'review_count >= {top_10_count} & avg_rating >= {avg_rating} & {similar}'
    interesting_query = f'review_count >= {avg_count} & avg_rating >= {avg_rating+0.5} & {similar}'

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

def get_titles(df, c=conn):
    """Returns titles of recommendations."""
    movies = list(df.index)
    SQL = """
            SELECT movieId, title FROM movies WHERE movieId IN({})
    """.format(','.join('?' * len(movies)))
    movies = pd.read_sql(SQL, c, params=movies)
    return pd.merge(df, movies, left_index=True, right_on='movieId')['title']

def get_movie(name, year=None, c=conn):
    """Returns movieId of named movie. Year can be provided for better accuracy in pull."""

    SQL = 'SELECT movieId, title FROM movies WHERE title LIKE ? LIMIT 1'
    if year:
        return pd.read_sql(SQL, c, params=(f'%{name}%({year})',))
    else:
        return pd.read_sql(SQL, c, params=(f'%{name}%',))

if __name__=="__main__":
    name = sys.argv[1]
    try:
        year = sys.argv[2]
    except:
        year = None

    try:
        selected_movie = get_movie(name, year).loc[0]
    except KeyError:
        print('Could not find movie')
        sys.exit()
    
    print(f"Recommendations based on: {selected_movie['title']}")
    print('-'*70)

    print('Processing')
    top_picks, interesting = recommendations(*preProcess(int(selected_movie['movieId'])))

    top_picks = get_titles(top_picks)
    interesting = get_titles(interesting)

    print('-'*70)
    print('Top Picks:')
    for title in top_picks.values:
        print(title)
    print('-'*70)
    print('Also Check Out:')
    for title in interesting.values:
        print(title)
    sys.exit()