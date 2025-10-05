#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.pipelines.frame_fetch.core import extractMovieFrames


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
    # Run the pipeline
    extractMovieFrames(configs)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
