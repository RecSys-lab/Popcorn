#!/usr/bin/env python3

from popcorn.utils import readConfigs, loadJsonFromUrl
from popcorn.datasets.popcorn.helper_metadata import (
    countMovies,
    fetchMovieById,
    fetchAllMovieIds,
    fetchRandomMovie,
    fetchMoviesByGenre,
    getAvgGenrePerMovie,
    fetchYearsOccurrences,
    fetchGenresOccurrences
)


def main():
    print(
        "Welcome to 'Popcorn' 🍿! Starting the framework for your movie recommendation ...\n"
    )
    # Read the configuration file
    configs = readConfigs("popcorn/config/config.yml")
    # If properly read, print the configurations
    if not configs:
        print("Error reading the configuration file!")
        return
    # Load Popcorn Dataset metadata
    datasetName = configs["datasets"]["multimodal"]["popcorn"]["name"]
    datasetMetadataUrl = configs["datasets"]["multimodal"]["popcorn"]["path_metadata"]
    print(
        f"- Loading the '{datasetName}' dataset metadata from '{datasetMetadataUrl}' ..."
    )
    jsonData = loadJsonFromUrl(datasetMetadataUrl)
    if jsonData is None:
        print("- Error in loading the Popcorn dataset metadata! Exiting ...")
        return
    # [Util-1] Count the number of movies in the dataset
    print("\n[Util-1] Counting the number of movies in the dataset ...")
    moviesCount = countMovies(jsonData)
    if moviesCount == -1:
        print("- Error in counting the number of movies!")
    else:
        print(f"- Number of movies in the dataset (from metadata): {moviesCount}")
    # [Util-2] Fetch all movie IDs in the dataset
    print("\n[Util-2] Fetching all movie IDs in the dataset ...")
    movieIds = fetchAllMovieIds(jsonData)
    if not movieIds:
        print("- Error in fetching movie IDs!")
    else:
        print(
            f"- {len(movieIds)} movie IDs have been fetched successfully. Sample IDs: {movieIds[:5]}"
        )
    # [Util-3] Fetch a random movie from the dataset
    print("\n[Util-3] Fetching a random movie from the dataset ...")
    randomMovie = fetchRandomMovie(jsonData)
    if not randomMovie:
        print("- Error in fetching a random movie!")
    else:
        print(f"- Random movie fetched successfully: {randomMovie}")
    # [Util-4] Fetch a movie by a given ID
    print("\n[Util-4] Fetching a movie by a given ID ...")
    givenMovieId = 6
    movieById = fetchMovieById(jsonData, givenMovieId)
    if not movieById:
        print(f"- Error in fetching movie by ID '{givenMovieId}'!")
    else:
        print(f"- Movie fetched successfully by ID '{givenMovieId}': {movieById}")
    # An unsuccessful ID fetch test
    unsuccessfulMovieId = 999999
    movieById = fetchMovieById(jsonData, unsuccessfulMovieId)
    if not movieById:
        print(f"- Error in fetching movie by ID '{unsuccessfulMovieId}'!")
    else:
        print(f"- Movie fetched successfully by ID '{unsuccessfulMovieId}': {movieById}")
    # [Util-5] Fetch movies by a given genre
    print("\n[Util-5] Fetching movies by a given genre ...")
    givenGenre = "Romance"
    moviesByGenre = fetchMoviesByGenre(jsonData, givenGenre)
    if not moviesByGenre:
        print(f"- Error in fetching movies by genre '{givenGenre}'!")
    else:
        print(f"- Sample movies fetched successfully by genre '{givenGenre}': {list(moviesByGenre.values())[:5]}")
    # [Util-6] Classify years by count
    print("\n[Util-6] Classifying years by count ...")
    yearsFreq = fetchYearsOccurrences(jsonData)
    if not yearsFreq:
        print("- Error in classifying years!")
    else:
        print(f"- Movies per year (based on metadata): {yearsFreq}")
    # [Util-7] Classify movies by genre
    print("\n[Util-7] Classifying movies by genre ...")
    genresFreq = fetchGenresOccurrences(jsonData)
    if not genresFreq:
        print("- Error in classifying movies by genre!")
    else:
        print(f"- Movies per genre (based on metadata): {genresFreq}")
    # [Util-8] Calculate the average genre per movie
    print("\n[Util-8] Calculating the average genre per movie ...")
    avgGenres = getAvgGenrePerMovie(jsonData)
    if avgGenres == 0.0:
        print("- Error in calculating the average genre per movie!")
    else:
        print(f"- Average genre per movie (based on metadata): {avgGenres}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
