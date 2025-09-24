#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.mmtf14k.helper_visual import loadVisualFusedDF


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
    # Load MMTF-14K dataset (visual fused - CNN)
    print("\n----------- MMTF-14K Visual (CNN) -----------")
    configs["datasets"]["multimodal"]["mmtf"]["visual_variant"] = "cnn"
    dfVisual = loadVisualFusedDF(configs)
    print(f"- Loaded visual fused DataFrame {dfVisual.shape} head:\n{dfVisual.head()}")
    # Load MMTF-14K dataset (visual fused - AVF)
    print("\n----------- MMTF-14K Visual (AVF) -----------")
    configs["datasets"]["multimodal"]["mmtf"]["visual_variant"] = "avf"
    dfVisual = loadVisualFusedDF(configs)
    print(f"- Loaded visual fused DataFrame {dfVisual.shape} head:\n{dfVisual.head()}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
