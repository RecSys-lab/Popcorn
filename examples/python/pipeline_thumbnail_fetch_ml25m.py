#!/usr/bin/env python3

import os
import shutil
from popcorn.utils import readConfigs
from popcorn.datasets.movielens.loader import loadMovieLens
from popcorn.pipelines.thumbnail_fetch.downloader import downloadThumbnails


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
    # Load MovieLens dataset - 25m version (only 25M has the linksDF)
    configs["datasets"]["unimodal"]["movielens"]["version"] = "25m"
    itemsDF, usersDF, ratingsDF, linksDF = loadMovieLens(configs)
    # Download thumbnails
    zipSize = 10 # 5000
    zipCounter = 1
    minItemId = 0
    maxItemId = 20 # len(itemsDF)
    for i in range(minItemId, maxItemId, zipSize):
        print(f"- Downloading items {i} to {i + zipSize} of {maxItemId} ...")
        # Prepare a movie list
        movieList = []
        for movie in itemsDF[i : i + zipSize].itertuples():
            movieList.append({"id": movie.item_id, "title": f'{movie.title}'})
        # Download thumbnails for the current batch of items
        downloadThumbnails(movieList, linksDF, configs)
        # Make a zip file
        rootFolder = configs["pipelines"]["thumbnail_fetch"]["download_path"]
        folderName = f"{rootFolder}/part{zipCounter}"
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        shutil.make_archive(folderName, "zip", folderName)
        zipCounter += 1


if __name__ == "__main__":
    main()
