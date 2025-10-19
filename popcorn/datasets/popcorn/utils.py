# URLs for accessing Popcorn dataset
BASE_URL = "https://huggingface.co/datasets/alitourani/Popcorn_Dataset"
RAW_DATA_URL = f"{BASE_URL}/raw/main/"
METADATA_URL = f"{BASE_URL}/resolve/main/stats.json"

# Supported CNN models
SUPPORTED_CNNS = ["incp3", "vgg19"]

# Supported aggregation models
AGG_MODELS = ["Max", "Mean"]

# Supported feature sources
EMBEDDING_SOURCES = ["full_movies", "movie_shots", "movie_trailers"]
AGG_EMBEDDING_SOURCES = ["full_movies_agg", "movie_shots_agg", "movie_trailers_agg"]


# Some checking functions
def isValidCNN(cnn: str) -> bool:
    cnn = cnn.lower()
    isValid = cnn in SUPPORTED_CNNS
    if not isValid:
        print(f"- [Error] Invalid CNN model '{cnn}'. Choose from {SUPPORTED_CNNS}.")
    return isValid


def isValidEmbeddingSource(source: str) -> bool:
    source = source.lower()
    isValid = source in EMBEDDING_SOURCES
    if not isValid:
        print(
            f"- [Error] Invalid embedding source '{source}'. Choose from {EMBEDDING_SOURCES}."
        )
    return isValid


def isValidAggEmbeddingSource(source: str) -> bool:
    source = source.lower()
    isValid = source in AGG_EMBEDDING_SOURCES
    if not isValid:
        print(
            f"- [Error] Invalid aggregated embedding source '{source}'. Choose from {AGG_EMBEDDING_SOURCES}."
        )
    return isValid


def isValidAggregationModel(model: str) -> bool:
    model = model.capitalize()
    isValid = model in AGG_MODELS
    if not isValid:
        print(
            f"- [Error] Invalid aggregation model '{model}'. Choose from {AGG_MODELS}."
        )
    return isValid
