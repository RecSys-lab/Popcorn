#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.popcorn.utils import RAW_DATA_URL
from popcorn.datasets.popcorn.helper_embedding_agg import (
    generateAggEmbeddingUrl,
    fetchAggregatedFeatures,
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
    # Load Popcorn Dataset metadata
    cnns = configs["datasets"]["multimodal"]["popcorn"]["cnns"]
    datasetName = configs["datasets"]["multimodal"]["popcorn"]["name"]
    aggEmbeddings = configs["datasets"]["multimodal"]["popcorn"]["agg_embedding_sources"]
    print(
        f"- Preparing to fetch the aggregated files of '{datasetName}' dataset from '{RAW_DATA_URL}' ..."
    )
    # [Util-1] Test generating sample address to aggregated features
    print(f"\n[Util-1] Generating a sample address to aggregated features ...")
    givenMovieId = 6
    givenCnn, givenEmbedding = cnns[0], aggEmbeddings[0]
    aggEmbeddingUrl = generateAggEmbeddingUrl(givenEmbedding, givenCnn, givenMovieId)
    print(
        f"- URL for aggregated features of movie '#{givenMovieId}' extracted by CNN '{givenCnn}' from source '{givenEmbedding}': {aggEmbeddingUrl}"
    )
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
