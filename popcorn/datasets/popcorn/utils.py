# URLs for accessing Popcorn dataset
BASE_URL = "https://huggingface.co/datasets/alitourani/Popcorn_Dataset"
RAW_DATA_URL = f"{BASE_URL}/raw/main/"
METADATA_URL = f"{BASE_URL}/resolve/main/stats.json"

# Supported CNN models
cnns = ["incp3", "vgg19"]

# Supported aggregation models
aggregationModels = ["Max", "Mean"]

# Supported feature sources
embeddingSources = ["full_movies", "movie_shots", "movie_trailers"]
aggEmbeddingSources = ["full_movies_agg", "movie_shots_agg", "movie_trailers_agg"]


# Some checking functions
def isValidCNN(cnn: str) -> bool:
    cnn = cnn.lower()
    isValid = cnn in cnns
    if not isValid:
        print(f"- [Error] Invalid CNN model '{cnn}'. Choose from {cnns}.")
    return isValid


def isValidEmbeddingSource(source: str) -> bool:
    source = source.lower()
    isValid = source in embeddingSources
    if not isValid:
        print(
            f"- [Error] Invalid embedding source '{source}'. Choose from {embeddingSources}."
        )
    return isValid


def isValidAggEmbeddingSource(source: str) -> bool:
    source = source.lower()
    isValid = source in aggEmbeddingSources
    if not isValid:
        print(
            f"- [Error] Invalid aggregated embedding source '{source}'. Choose from {aggEmbeddingSources}."
        )
    return isValid


def isValidAggregationModel(model: str) -> bool:
    model = model.capitalize()
    isValid = model in aggregationModels
    if not isValid:
        print(
            f"- [Error] Invalid aggregation model '{model}'. Choose from {aggregationModels}."
        )
    return isValid
