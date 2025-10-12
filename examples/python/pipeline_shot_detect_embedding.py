#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.pipelines.shot_detect.frame_based import extractShotsFromFrames
from popcorn.pipelines.shot_detect.embedding_based import extractShotsFromEmbeddings


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
    # Shot detection from embeddings
    extractShotsFromEmbeddings(configs)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
