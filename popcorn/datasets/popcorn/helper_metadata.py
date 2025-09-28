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
        print(
            "- [Warn] Metadata is empty or not loaded. Returning -1 as movie count ..."
        )
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
        movieIds = [movie["id"] for movie in data]
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty list of movie IDs ..."
        )
    # Return the list of movie IDs
    return movieIds


def fetchRandomMovie(data: dict) -> dict:
    """
    Fetches a random movie from the given metadata JSON file.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.

    Returns
    -------
    randomMovie: dict
        A dictionary containing the metadata of a random movie.
    """
    # Variables
    randomMovie = {}
    # Fetch a random movie
    if data:
        randomMovie = random.choice(data)
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty dictionary as random movie ..."
        )
    # Return the random movie
    return randomMovie

def fetchRandomMovies(data: dict, count: int = 5) -> list:
    """
    Fetches a list of random movies from the given metadata JSON file.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.
    count: int
        The number of random movies to fetch.

    Returns
    -------
    randomMovies: list
        A list of dictionaries, each containing the metadata of a random movie.
    """
    # Variables
    randomMovies = []
    # Check validity of count
    if count <= 0:
        print("- [Warn] Count must be a positive integer. Returning an empty list of random movies ...")
        return randomMovies
    if count > len(data):
        print(f"- [Warn] Count '{count}' exceeds the number of available movies '{len(data)}'. Reducing count to '5'.")
        count = 5
    # Fetch random movies
    if data:
        randomMovies = random.sample(data, count)
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty list of random movies ..."
        )
    return randomMovies


def fetchMovieById(data: dict, movieId: int) -> dict:
    """
    Fetches a movie by its ID from the given metadata JSON file.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.
    movieId: int
        The ID of the movie to fetch.

    Returns
    -------
    movie: dict
        A dictionary containing the metadata of the movie with the given ID.
    """
    # Variables
    movie = {}
    # Fetch the movie by ID
    if data:
        # Standardize movieId to 10 digits
        standardizedId = f"{int(movieId):010d}"
        # Find the movie with the given ID
        for movieData in data:
            if movieData.get("id") == standardizedId:
                movie = movieData
                break
        # If no movie is found with the given ID
        if not movie:
            print(
                f"- [Warn] No movie found with the given ID '{standardizedId}'. Returning an empty dictionary as movie ..."
            )
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty dictionary as movie ..."
        )
    # Return the movie (empty if not found)
    return movie


def fetchMoviesByGenre(data: dict, genre: str) -> dict:
    """
    Fetch movies by a single genre from the given data.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.
    genre: str
        The genre to filter the movies by.

    Returns
    -------
    matchedMovies: dict
        A dictionary containing the movies that match the given genre.
    """
    # Variables
    matchedMovies = {}
    # Fetch movies by genre
    if data:
        # Find movies that match the given genre
        matchedMovies = {
            movie["id"]: movie for movie in data if genre in movie.get("genres", [])
        }
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty dictionary as matched movies ..."
        )
    # Return the matched movies
    return matchedMovies


def fetchYearsOccurrences(data: dict) -> dict:
    """
    Classify all the years in the dataset by count.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.

    Returns
    -------
    yearsFreq: dict
        A dictionary containing the years as keys and their counts as values.
    """
    # Variables
    yearsFreq = {}
    # Classify years by count
    if data:
        # Extract years
        years = [movie["year"] for movie in data if "year" in movie]
        # Use Counter to count occurrences of each year
        yearsCount = Counter(years)
        # Convert Counter object to a regular dictionary
        yearsFreq = dict(yearsCount)
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty dictionary as classified years ..."
        )
    # Return the classified years
    return yearsFreq


def fetchGenresOccurrences(data: dict) -> dict:
    """
    Classify all the movies in the dataset by genre.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.

    Returns
    -------
    genresFreq: dict
        A dictionary containing the genres as keys and their counts as values.
    """
    # Variables
    genresFreq = {}
    genresCount = Counter()
    # Classify movies by genre
    if data:
        for movie in data:
            # Extract genres from each movie
            genres = movie.get("genres", [])
            # Update the Counter with the genres
            genresCount.update(genres)
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning an empty dictionary as classified genres ..."
        )
    # Return the classified genres
    genresFreq = dict(genresCount)
    return genresFreq


def getAvgGenrePerMovie(data: dict) -> float:
    """
    Calculate the average number of genres per movie.

    Parameters
    ----------
    data: dict
        The JSON data containing the metadata of the movies.
    
    Returns
    -------
    avgGenres: float
        The average number of genres per movie.
    """
    # Variables
    avgGenres = 0.0
    # Check if the genres dictionary is not empty
    if data:
        # Classify movies by genre
        moviesByGenre = fetchGenresOccurrences(data)
        # Count the number of movies
        moviesCount = countMovies(data)
        # Calculate the total number of genres
        totalGenres = sum(moviesByGenre.values())
        # Calculate the average number of genres per movie
        avgGenres = round(totalGenres / moviesCount, 3)
    else:
        print(
            "- [Warn] Metadata is empty or not loaded. Returning 0.0 as average genres per movie ..."
        )
    # Return the average genres (0.0 if not calculated)
    return avgGenres
