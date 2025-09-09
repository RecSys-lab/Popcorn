#!/usr/bin/env python3

from popcorn.utils import readConfigs
from popcorn.datasets.poison_rag_plus.loader import loadPoisonRagPlus


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
    # Load Poison-RAG-Plus LLM Augmented Llama
    print("\n----------- Llama + Augmented -----------")
    configs["datasets"]["unimodal"]["poison_rag_plus"]["llm"] = "llama"
    configs["datasets"]["unimodal"]["poison_rag_plus"]["augmented"] = True
    itemsTextDF = loadPoisonRagPlus(configs)
    if itemsTextDF is not None:
        print(f"\n- itemsTextDF (shape: {itemsTextDF.shape}): \n{itemsTextDF.head()}")
    # Load Poison-RAG-Plus LLM Original Llama
    print("\n----------- Llama + Original -----------")
    configs["datasets"]["unimodal"]["poison_rag_plus"]["augmented"] = False
    itemsTextDF = loadPoisonRagPlus(configs)
    if itemsTextDF is not None:
        print(f"\n- itemsTextDF (shape: {itemsTextDF.shape}): \n{itemsTextDF.head()}")
    # Load Poison-RAG-Plus LLM Original OpenAI
    print("\n----------- OpenAI + Original -----------")
    configs["datasets"]["unimodal"]["poison_rag_plus"]["llm"] = "openai"
    itemsTextDF = loadPoisonRagPlus(configs)
    if itemsTextDF is not None:
        print(f"\n- itemsTextDF (shape: {itemsTextDF.shape}): \n{itemsTextDF.head()}")
    # Load Poison-RAG-Plus LLM Augmented OpenAI
    print("\n----------- OpenAI + Augmented -----------")
    configs["datasets"]["unimodal"]["poison_rag_plus"]["augmented"] = True
    itemsTextDF = loadPoisonRagPlus(configs)
    if itemsTextDF is not None:
        print(f"\n- itemsTextDF (shape: {itemsTextDF.shape}): \n{itemsTextDF.head()}")
    # Load Poison-RAG-Plus LLM Augmented SentenceTransformer
    print("\n----------- SentenceTransformer + Augmented -----------")
    configs["datasets"]["unimodal"]["poison_rag_plus"]["llm"] = "st"
    itemsTextDF = loadPoisonRagPlus(configs)
    if itemsTextDF is not None:
        print(f"\n- itemsTextDF (shape: {itemsTextDF.shape}): \n{itemsTextDF.head()}")
    # Load Poison-RAG-Plus LLM Original SentenceTransformer
    print("\n----------- SentenceTransformer + Original -----------")
    configs["datasets"]["unimodal"]["poison_rag_plus"]["augmented"] = False
    itemsTextDF = loadPoisonRagPlus(configs)
    if itemsTextDF is not None:
        print(f"\n- itemsTextDF (shape: {itemsTextDF.shape}): \n{itemsTextDF.head()}")
    # Stop
    print("\nStopping 'Popcorn'!")


if __name__ == "__main__":
    main()
