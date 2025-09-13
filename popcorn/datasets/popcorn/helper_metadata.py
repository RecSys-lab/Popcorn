import random
from collections import Counter


def countMovies(data: dict) -> int:
    """
    Counts the number of movies in the given metadata JSON file.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.

    Returns
    -------
    moviesCount: int
        The number of movies in the dataset.
    """
    # Variables
    moviesCount = -1
    # Count the number of movies
    if data:
        moviesCount = len(data)
    else:
        print("- [Warn] Metadata is empty or not loaded.")
    # Return the count of movies
    return moviesCount

def fetchAllMovieIds(data: dict) -> list:
    """
    Fetches all the movie IDs from the given metadata JSON file.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.

    Returns
    -------
    movieIds: list
        A list of all movie IDs in the dataset.
    """
    # Variables
    movieIds = []
    # Fetch all movie IDs
    if data:
        movieIds = [movie['id'] for movie in data]
    else:
        print("- [Warn] Metadata is empty or not loaded.")
    # Return the list of movie IDs
    return movieIds

def fetchRandomMovie(data):
    """
    Fetches a random movie from the given data.

    Parameters:
        data (dict): The JSON data containing the movies.

    Returns:
        dict: A random movie from the dataset.
    """
    if data:
        randomMovie = random.choice(data)
        # print("Randomly fetched movie:")
        # print(json.dumps(randomMovie, indent=4))
        return randomMovie
    else:
        print("Data is empty or not loaded.")
        return {}

def fetchMovieById(data, movieId):
    """
    Fetches a movie by its ID from the given data.

    Parameters:
        data (dict): The JSON data containing the movies.
        movieId (int): The ID of the movie to fetch.
    """
    if data:
        # Standardize movieId to 10 digits
        standardizedId = f"{int(movieId):010d}"
        # Find the movie with the given ID
        for movie in data:
            if movie.get('id') == standardizedId:
                # print("Fetched movie by ID:")
                # print(json.dumps(movie, indent=4))
                return movie
        # If no movie is found with the given ID
        print(f"No movie found with ID: {standardizedId}")
    else:
        print("Data is empty or not loaded.")
        return {}

def fetchMoviesByGenre(data, genre):
    """
    Fetch movies by a single genre from the given data.

    Parameters:
        data (dict): The JSON data containing the movies.
        genre (str): The genre to filter the movies by.

    Returns:
        dict: A dictionary containing the matched movies.
    """
    matchedMovies = {}
    if data:
        matchedMovies = {movie['id']: movie for movie in data if genre in movie.get('genres', [])}
    return matchedMovies

def classifyYearsByCount(data):
    """
    Classify all the years in the dataset by count.

    Parameters:
        data (dict): The JSON data containing the movies.

    Returns:
        dict: A dictionary containing the years as keys and their counts as values.
    """
    if data:
        years = [movie['year'] for movie in data if 'year' in movie]
        yearsCount = Counter(years)
        return dict(yearsCount)
    else:
        print("Data is empty or not loaded.")
        return {}

def calculateAverageGenrePerMovie(genresDict, moviesCount):
    """
    Calculate the average number of genres per movie.

    Parameters:
        genresDict (dict): A dictionary containing genres as keys and their counts as values.
        moviesCount (int): The total number of movies in the dataset.
        
    Returns:
        float: The average number of genres per movie.
    """
    # Check if the genres dictionary is not empty
    if genresDict:
        # Calculate the total number of genres
        totalGenres = sum(genresDict.values())
        # Calculate the average number of genres per movie
        averageGenrePerMovie = round(totalGenres / moviesCount, 3)
        # Return the result
        return averageGenrePerMovie
    else:
        print("Genres dictionary is empty!")

def classifyMoviesByGenre(data):
    """
    Classify all the movies in the dataset by genre.

    Parameters:
        data (dict): The JSON data containing the movies.

    Returns:
        dict: A dictionary containing the genres as keys and their counts as values.
    """
    genreCounts = Counter()
    if data:
        for movie in data:
            genres = movie.get('genres', [])
            genreCounts.update(genres)
    return dict(genreCounts)