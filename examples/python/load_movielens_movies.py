#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.datasets.movielens.helper_movies import filterMoviesByGenre
from popcorn.datasets.movielens.helper_movies import filterMoviesWithMainGenres


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
    # Load MovieLens dataset - 1m version
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if itemsDF is None:
        print("Error in loading the MovieLens dataset! Exiting ...")
        return
    print(f"\n- ItemsDF (shape: {itemsDF.shape}): \n{itemsDF.head()}")
    # [Util-1] Filter movies by a given genre
    print("\n[Util-1] Filtering movies by given genres ...")
    itemsDF_filtered = filterMoviesByGenre(itemsDF, genre="Comedy")
    print(f"- Filtered ItemsDF: \n{itemsDF_filtered.head(3)}\n")
    itemsDF_filtered = filterMoviesByGenre(itemsDF, genre="Drama")
    print(f"- Filtered ItemsDF: \n{itemsDF_filtered.head(3)}\n")
    itemsDF_filtered = filterMoviesByGenre(itemsDF, genre="TestGenre")
    # [Util-2] Filter movies containing the main genres
    print("\n[Util-2] Filtering movies containing the main genres ...")
    itemsDF_mainGenres = filterMoviesWithMainGenres(itemsDF)
    print(
        f"- Main Genres ItemsDF (shape: {itemsDF_mainGenres.shape}): \n{itemsDF_mainGenres.head(3)}"
    )
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
