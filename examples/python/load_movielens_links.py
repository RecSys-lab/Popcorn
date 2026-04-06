#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.movielens.loader import loadMovieLens


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
    if linksDF is None:
        print("Error in loading the MovieLens dataset! Exiting ...")
        return
    print(f"\n- linksDF (shape: {linksDF.shape}): \n{linksDF.head(5)}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
