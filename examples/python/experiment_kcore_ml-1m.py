#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.utils import applyKcore
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
    # Load MovieLens dataset - 1m version
    configs["datasets"]["unimodal"]["movielens"]["version"] = "1m"
    itemsDF, usersDF, ratingsDF = loadMovieLens(configs)
    if ratingsDF is not None:
        print(f"\n- RatingsDF original row count: {len(ratingsDF):,}")
    # Now, apply k-core filtering and see the difference
    K_CORE = configs["setup"]["k_core"]
    ratingsDF_filtered = applyKcore(ratingsDF, K_CORE)
    print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF_filtered):,}")
    # Different k-core values: 20
    ratingsDF_filtered = applyKcore(ratingsDF, 20)
    print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF_filtered):,}")
    # Different k-core values: 40
    ratingsDF_filtered = applyKcore(ratingsDF, 40)
    print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF_filtered):,}")
    # Different k-core values: 60
    ratingsDF_filtered = applyKcore(ratingsDF, 60)
    print(f"- After {K_CORE}-core filtering row count: {len(ratingsDF_filtered):,}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
