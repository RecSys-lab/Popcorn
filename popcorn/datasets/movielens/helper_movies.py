#!/usr/bin/env python3

import pandas as pd

mainGenres = ['Action', 'Comedy', 'Drama', 'Horror']

def fetchAllUniqueGenres(dataFrame: pd.DataFrame):
    """
    Counts the number of movies in the given data.

    Parameters:
    ----------
    dataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    
    Returns:
    -------
    allGenres: list
        A list of all genres in the dataset.
    """
    # Split the 'genres' column by '|' and explode the list into rows
    allGenres = dataFrame['genres'].str.split('|').explode()
    # Get the unique genres
    uniqueGenres = allGenres.unique()    
    # Return unique genres
    return uniqueGenres

def fetchMoviesByGenre(dataFrame: pd.DataFrame, genre: str):
    """
    Fetches movies by a given genre.

    Parameters:
    ----------
    dataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    genre: str
        The genre to filter the movies by.
    
    Returns:
    -------
    moviesByGenre: pd.DataFrame
        The DataFrame containing the movies of the given genre.
    """
    # Filter the DataFrame by the given genre
    moviesByGenre = dataFrame[dataFrame['genres'].str.contains(genre)]
    # Return the filtered DataFrame
    return moviesByGenre

def filterMoviesWithMainGenres(dataFrame: pd.DataFrame):
    """
    Filters movies by the main genres.

    Parameters:
    ----------
    dataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    
    Returns:
    -------
    mainGenresMovies: pd.DataFrame
        The DataFrame containing the movies of the main genres.
    """
    # Filter the DataFrame by the main genres
    mainGenresMovies = dataFrame[dataFrame['genres'].str.contains('|'.join(mainGenres))]
    # Return the filtered DataFrame
    return mainGenresMovies

def binarizeMovieGenres(dataFrame: pd.DataFrame):
    """
    Binarizes the genres column based on the presence of main genres.

    Parameters:
    ----------
    dataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    
    Returns:
    -------
    moviesDFBinarizedGenres: pd.DataFrame
        A DataFrame with movieId and separate columns for each main genre, indicating presence (1) or absence (0).
    """
    # Variables
    moviesDFBinarizedGenres = pd.DataFrame(columns=['movieId', 'isAction', 'isComedy', 'isDrama', 'isHorror'])
    # Iterate over the main genres
    for genre in mainGenres:
        # Create a new column for each genre
        moviesDFBinarizedGenres[f'is{genre}'] = dataFrame['genres'].str.contains(genre).astype(int)
    # Add the movieId column
    moviesDFBinarizedGenres['movieId'] = dataFrame['movieId']
    # Return the binarized DataFrame
    return moviesDFBinarizedGenres

def augmentMoviesDFWithBinarizedGenres(moviesDataFrame: pd.DataFrame, binarizedGenresDataFrame: pd.DataFrame):
    """
    Augments the movies DataFrame with binarized genres.

    Parameters:
    ----------
    moviesDataFrame: pd.DataFrame
        The DataFrame containing the movie data.
    binarizedGenresDataFrame: pd.DataFrame
        The DataFrame containing the binarized genres.
    
    Returns:
    -------
    augmentedMoviesDataFrame: pd.DataFrame
        The augmented DataFrame containing the movie data with binarized genres.
    """
    # Merge the movies DataFrame with the binarized genres DataFrame
    augmentedMoviesDataFrame = pd.merge(moviesDataFrame, binarizedGenresDataFrame, on='movieId')
    # Return the augmented DataFrame
    return augmentedMoviesDataFrame

def loadGenres(download_path_prefix: str, DATASET: str) -> pd.DataFrame:
    """
    Load genres from the MovieLens dataset.

    Parameters
    ----------
    download_path_prefix : str
        The prefix path where the dataset files are downloaded.
    DATASET : str
        The version of the MovieLens dataset ('100k' or '1m').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their associated genres.
    """
    if DATASET == "100k":
        # Variables
        genre_cols = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        cols = [
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
        ] + genre_cols
        dest_data = os.path.join(download_path_prefix, "u.item")
        # Read the movies file
        movies = pd.read_csv(
            dest_data, sep="|", header=None, names=cols, encoding="latin-1"
        )
        movies["genres"] = movies[genre_cols].apply(
            lambda row: [g for g in genre_cols if row[g] == 1], axis=1
        )
        movies["item_id"] = movies["item_id"].astype(str)
    else:
        # Variables
        dest_folder = os.path.join(download_path_prefix, "ml-1m")
        path = os.path.join(dest_folder, "ml-1m/movies.dat")
        if not os.path.exists(path):
            path = os.path.join(dest_folder, "ml-1m/ml-1m/movies.dat")
        # Read the movies file
        movies = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )
        movies["item_id"] = movies["item_id"].astype(str)
        movies["genres"] = movies["genres"].map(
            lambda s: s.split("|") if isinstance(s, str) else []
        )
    return movies[["item_id", "genres"]]