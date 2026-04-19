#!/usr/bin/env python3

import pandas as pd
from popcorn.utils import readConfigs
from popcorn.datasets.ml_thumbnail.helper_embedding import (
    loadMovieLensThumbnailEmbeddings,
    loadAllMovieLensThumbnailEmbeddings
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
    # Load thumbnail embeddings for a given part
    configs["datasets"]["unimodal"]["ml_thumbnail"]["variant"] = 'dino-v2'
    print("\n----------- Thumbnail Embedding Loading -----------")
    partId = 1
    variant = configs["datasets"]["unimodal"]["ml_thumbnail"]["variant"]
    embeddingsDF = loadMovieLensThumbnailEmbeddings(partId, variant)
    if embeddingsDF is not None:
        print(f"Loaded thumbnail embeddings for part {partId}, variant '{variant}'")
    # Load all thumbnail embeddings
    print("\n----------- All Thumbnail Embeddings Loading -----------")
    allEmbeddingsDF = loadAllMovieLensThumbnailEmbeddings(variant)
    if allEmbeddingsDF is not None:
        print(f"Loaded all thumbnail embeddings")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
