#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.mmtf14k.helper_audio import loadAudioFusedDF


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
    # Load MMTF-14K dataset (audio fused - i-vector)
    print("\n----------- MMTF-14K Audio (i-vector) -----------")
    configs["datasets"]["multimodal"]["mmtf"]["audio_variant"] = "ivec"
    dfAudio = loadAudioFusedDF(configs)
    print(f"- Loaded audio fused DataFrame {dfAudio.shape} head:\n{dfAudio.head()}")
    # Load MMTF-14K dataset (audio fused - blf)
    print("\n----------- MMTF-14K Audio (blf) -----------")
    configs["datasets"]["multimodal"]["mmtf"]["audio_variant"] = "blf"
    dfAudio = loadAudioFusedDF(configs)
    print(f"- Loaded audio fused DataFrame {dfAudio.shape} head:\n{dfAudio.head()}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
