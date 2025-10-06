#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.pipelines.visual_embedding.aggregation import aggregateVisualEmbeddings


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
    # Run the visual embedding extraction pipeline
    aggregateVisualEmbeddings(configs)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
