#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.popcorn.utils import RAW_DATA_URL
from popcorn.datasets.popcorn.helper_embedding_agg import (
    loadAggEmbeddings,
    fetchMovieAggEmbeddings,
    generateAllAggEmbeddingUrls,
    generateMovieAggEmbeddingUrl,
    generateMoviesAggEmbeddingUrls,
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
    aggEmbeddingUrl = generateMovieAggEmbeddingUrl(givenEmbedding, givenCnn, givenMovieId)
    print(
        f"- URL for aggregated features of movie '#{givenMovieId}' extracted by CNN '{givenCnn}' from source '{givenEmbedding}': {aggEmbeddingUrl}"
    )
    # [Util-2] Test fetching aggregated features of a movie
    print(f"\n[Util-2] Fetching aggregated features of a movie ...")
    aggEmbeddingList = fetchMovieAggEmbeddings(givenEmbedding, givenCnn, givenMovieId)
    print(f"- Fetched {len(aggEmbeddingList)} aggregated features! Sample embeddings:")
    print(f"-- 'Max': {aggEmbeddingList[0]['Max'][:5]}")
    print(f"-- 'Mean': {aggEmbeddingList[0]['Mean'][:5]}")
    # [Util-3] Test generating all addresses to aggregated features
    givenMovieIds = [6, 50]
    aggEmbeddings = ["full_movies_agg", "movie_trailers_agg"]
    print(f"\n[Util-3] Generating all addresses to aggregated features for movies {givenMovieIds} extracted by CNNs {cnns} from sources {aggEmbeddings} ...")
    allAggEmbeddingUrls = generateMoviesAggEmbeddingUrls(aggEmbeddings, cnns, givenMovieIds)
    print(f"- Generated {len(allAggEmbeddingUrls)} variants of aggregated feature addresses!")
    print(f"-- Keys: {list(allAggEmbeddingUrls.keys())}")
    print(f"-- Sample addresses for '{aggEmbeddings[0]}' and '{cnns[0]}': {allAggEmbeddingUrls[aggEmbeddings[0]][cnns[0]]}")
    # [Util-4] Test generating all addresses to aggregated features based on the configuration file
    print(f"\n[Util-4] Generating all addresses to aggregated features based on the configuration file ...")
    aggEmbeddingUrlDict = generateAllAggEmbeddingUrls(configs)
    if not aggEmbeddingUrlDict:
        print("- Error in generating the addresses! Continuing ...")
    else:
        print(f"- Generated {len(aggEmbeddingUrlDict)} variants of aggregated feature addresses!")
        print(f"- Sample addresses for '{aggEmbeddings[0]}' and '{cnns[0]}': {aggEmbeddingUrlDict[aggEmbeddings[0]][cnns[0]][:2]}")
    # [Util-5] Test loading aggregated features into a DataFrame
    print(f"\n[Util-5] Loading aggregated features into a DataFrame ...")
    if not aggEmbeddingUrlDict:
        print("- Error in loading the aggregated features! Skipping ...")
    else:
        # Take a few samples of the generated addresses
        aggEmbeddingUrlList = aggEmbeddingUrlDict[aggEmbeddings[0]][cnns[0]][:5]
        # Load aggregated features into a DataFrame
        dfAggEmbedsMax, dfAggEmbedsMean = loadAggEmbeddings(aggEmbeddingUrlList)
        print(f"- Loaded {len(dfAggEmbedsMax)} sample records of aggregated features! Check the first 3 records:")
        print(f"- The loaded DataFrame (Max):\n{dfAggEmbedsMax.head(3)}")
        print(f"- The loaded DataFrame (Mean):\n{dfAggEmbedsMean.head(3)}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
