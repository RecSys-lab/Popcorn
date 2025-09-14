#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.popcorn.utils import RAW_DATA_URL
from popcorn.datasets.popcorn.helper_embedding import generatePacketUrl, fetchAllPackets


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
    embeddings = configs["datasets"]["multimodal"]["popcorn"]["embedding_sources"]
    print(
        f"- Preparing to fetch the raw file of '{datasetName}' dataset from '{RAW_DATA_URL}' ..."
    )
    # [Util-1] Test generating sample packet URLs
    print(f"\n[Util-1] Generating a sample packet URLs to embeddings ...")
    givenMovieId, givenPacketId = 6, 1
    givenCnn, givenEmbedding = cnns[0], embeddings[0]
    packetUrl = generatePacketUrl(givenEmbedding, givenCnn, givenMovieId, givenPacketId)
    print(
        f"- URL for packet '#{givenPacketId}' of movie '#{givenMovieId}' extracted by CNN '{givenCnn}' from source '{givenEmbedding}': {packetUrl}"
    )
    # Another sample
    givenMovieId, givenPacketId = 150, 3
    givenCnn, givenEmbedding = "vgg19", "movie_trailers"
    packetUrl = generatePacketUrl(givenEmbedding, givenCnn, givenMovieId, givenPacketId)
    print(
        f"- URL for packet '#{givenPacketId}' of movie '#{givenMovieId}' extracted by CNN '{givenCnn}' from source '{givenEmbedding}': {packetUrl}"
    )
    # [Util-2] Test fetching all packets of a movie
    print(f"\n[Util-2] Fetching all packets of a movie ...")
    givenMovieId = 6
    givenCnn, givenEmbedding = "incp3", "movie_trailers"
    fetchedEmbeddings = fetchAllPackets(givenEmbedding, givenCnn, givenMovieId)
    print(
        f"- Number of embeddings from the fetched packets for movie '#{givenMovieId}': {len(fetchedEmbeddings)}"
    )
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
