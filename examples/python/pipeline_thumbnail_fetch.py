#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.pipelines.thumbnail_fetch.core import getMovieList
from popcorn.pipelines.thumbnail_fetch.downloader import downloadThumbnails


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
    # Load MovieLens dataset - 25m version (only 25M has the linksDF)
    configs["datasets"]["unimodal"]["movielens"]["version"] = "25m"
    itemsDF, usersDF, ratingsDF, linksDF = loadMovieLens(configs)
    # Get the list of movies - Dummy dataset variant (for testing)
    print("\n----------- Thumbnail Download - Dummy -----------")
    datasetVariant = "dummy"
    configs["pipelines"]["thumbnail_fetch"]["dataset_variant"] = datasetVariant
    movieList = getMovieList(datasetVariant, configs)
    downloadThumbnails(movieList, linksDF, configs)
    # Get the list of movies - MMTF dataset variant
    print("\n----------- Thumbnail Download - MMTF -----------")
    datasetVariant = "mmtf"
    configs["pipelines"]["thumbnail_fetch"]["dataset_variant"] = datasetVariant
    movieList = getMovieList(datasetVariant, configs)
    downloadThumbnails(movieList, linksDF, configs)
    print("\nStopping 'Popcorn'!")
    # Get the list of movies - Popcorn dataset variant
    print("\n----------- Thumbnail Download - Popcorn -----------")
    datasetVariant = "popcorn"
    configs["pipelines"]["thumbnail_fetch"]["dataset_variant"] = datasetVariant
    movieList = getMovieList(datasetVariant, configs)
    downloadThumbnails(movieList, linksDF, configs)


if __name__ == "__main__":
    main()
