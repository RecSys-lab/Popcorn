#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.utils import applyKcore
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.datasets.movielens.process import trainTestSplit


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
    # Some example configurations
    configs["setup"]["k_core"] = 40
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"
    # Load MovieLens dataset - 1m version
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if ratingsDF is None:
        print("Error in loading the MovieLens dataset! Exiting ...")
        return
    print(f"\n- RatingsDF original row count: {len(ratingsDF):,}")
    # Now, apply k-core filtering and see the difference
    K_CORE = configs["setup"]["k_core"]
    ratingsDF_filtered = applyKcore(ratingsDF, K_CORE)
    print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF_filtered):,}")
    # I. Split the data into train and test sets ('random' mode with 20% test ratio)
    configs["setup"]["split"]["mode"] = "random"
    configs["setup"]["split"]["test_ratio"] = 0.2
    trainTestSplit(ratingsDF_filtered, configs)
    # II. Split the data into train and test sets ('temporal' mode with 25% test ratio)
    configs["setup"]["split"]["mode"] = "temporal"
    configs["setup"]["split"]["test_ratio"] = 0.25
    trainTestSplit(ratingsDF_filtered, configs)
    # III. Split the data into train and test sets ('per_user' mode with 30% test ratio)
    configs["setup"]["split"]["mode"] = "per_user"
    configs["setup"]["split"]["test_ratio"] = 0.3
    trainTestSplit(ratingsDF_filtered, configs)
    # IV. Split the data into train and test sets with unknown mode (to see the error handling)
    configs["setup"]["split"]["mode"] = "leave_one_out"
    trainTestSplit(ratingsDF_filtered, configs)
    # V. Split the data into train and test sets with invalid test ratio (to see the error handling)
    configs["setup"]["split"]["mode"] = "random"
    configs["setup"]["split"]["test_ratio"] = 1.5
    trainTestSplit(ratingsDF_filtered, configs)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
