#!/usr/bin/env python3

from popcorn.utils import readConfigs

def main():
    print("Welcome to 'Popcorn' 🍿! Starting the framework for your movie recommendation ...\n")
    # Read the configuration file
    configs = readConfigs("popcorn/config/config.yml")
    # If properly read, print the configurations
    if configs:
        # Get common configurations
        cfgGeneral = configs['general']
        cfgDatasets = configs['datasets']
        cfgPipeline = configs['pipelines']
        print(f"\n- General configurations: {cfgGeneral}")
        print(f"\n- Datasets configurations: {cfgDatasets}")
        print(f"\n- Pipelines configurations: {cfgPipeline}")
    print("\nStopping 'Popcorn'!")

if __name__ == "__main__":
    main()