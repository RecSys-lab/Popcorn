#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.pipelines.visual_embedding.extraction import extractVisualEmbeddings


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
    # Use VGG19 model to extract visual embeddings
    print("\n----------- VGG19 -----------")
    configs["pipelines"]["visual_embedding_extractor"]["cnn"] = "vgg19"
    extractVisualEmbeddings(configs)
    # Use InceptionV3 model to extract visual embeddings
    print("\n----------- InceptionV3 -----------")
    configs["pipelines"]["visual_embedding_extractor"]["cnn"] = "incp3"
    extractVisualEmbeddings(configs)
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
