#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.datasets.movielens.helper_movies import augmentMoviesWithBinarizedGenres
from popcorn.datasets.movielens.helper_genres import (
    binarizeGenres,
    getGenreDict,
    getMainGenres,
    getAllGenres,
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
    # Load MovieLens dataset - 100k version
    configs["datasets"]["unimodal"]["movielens"]["version"] = "100k"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if itemsDF is None:
        print("Error in loading the MovieLens dataset! Exiting ...")
        return
    print(f"\n- ItemsDF (shape: {itemsDF.shape}): \n{itemsDF.head()}")
    # [Util-1] Get main and all unique genres
    print("\n[Util-1] Getting main and all unique genres ...")
    print(f"- Main Genres: {getMainGenres()}")
    print(f"- All Genres: {getAllGenres()}")
    # [Util-2] Get genre dictionary and save genres DataFrame
    print("\n[Util-2] Getting genre dictionary and saving genres DataFrame ...")
    genreDict = getGenreDict(itemsDF, configs, saveOutput=True)
    print(f"- Genre Dictionary (sample): \n{dict(list(genreDict.items())[:3])}")
    # [Util-3] Binarize genres as a new ItemsDF
    print("\n[Util-3] Binarizing genres as a new ItemsDF ...")
    itemsDF_binGenre = binarizeGenres(itemsDF)
    print(f"- ItemsDF with binarized genres: \n{itemsDF_binGenre.head(3)}")
    # [Util-4] Augment original ItemsDF with binarized genres
    print("\n[Util-4] Augmenting original ItemsDF with binarized genres ...")
    itemsDF_augmented = augmentMoviesWithBinarizedGenres(itemsDF, itemsDF_binGenre)
    print(f"- Augmented ItemsDF: \n{itemsDF_augmented.head(3)}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
