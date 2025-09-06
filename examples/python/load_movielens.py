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
    # Load MovieLens dataset - 100k version
    print("\n----------- MovieLens 100k -----------")
    configs["datasets"]["unimodal"]["movielens"]["version"] = "100k"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if itemsDF is not None:
        print(f"\n- ItemsDF (shape: {itemsDF.shape}): \n{itemsDF.head()}")
    if usersDF is not None:
        print(f"\n- UsersDF (shape: {usersDF.shape}): \n{usersDF.head()}")
    if ratingsDF is not None:
        print(f"\n- RatingsDF (shape: {ratingsDF.shape}): \n{ratingsDF.head()}")
    # Load MovieLens dataset - 1m version
    print("\n----------- MovieLens 1m -----------")
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if itemsDF is not None:
        print(f"\n- ItemsDF (shape: {itemsDF.shape}): \n{itemsDF.head()}")
    if usersDF is not None:
        print(f"\n- UsersDF (shape: {usersDF.shape}): \n{usersDF.head()}")
    if ratingsDF is not None:
        print(f"\n- RatingsDF (shape: {ratingsDF.shape}): \n{ratingsDF.head()}")
    # Load MovieLens dataset - 25m version
    print("\n----------- MovieLens 25m -----------")
    configs["datasets"]["unimodal"]["movielens"]["version"] = "25m"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if itemsDF is not None:
        print(f"\n- ItemsDF (shape: {itemsDF.shape}): \n{itemsDF.head()}")
    if usersDF is not None:
        print(f"\n- UsersDF (shape: {usersDF.shape}): \n{usersDF.head()}")
    if ratingsDF is not None:
        print(f"\n- RatingsDF (shape: {ratingsDF.shape}): \n{ratingsDF.head()}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
