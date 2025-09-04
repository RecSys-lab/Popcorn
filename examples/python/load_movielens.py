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
    # Download MovieLens dataset - 100k version
    configs['datasets']['unimodal']['movielens']["version"] = "100k"
    loadMovieLens(configs)
    # Download MovieLens dataset - 1m version
    configs['datasets']['unimodal']['movielens']["version"] = "1m"
    loadMovieLens(configs)
    # Download MovieLens dataset - 25m version
    configs['datasets']['unimodal']['movielens']["version"] = "25m"
    loadMovieLens(configs)
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
