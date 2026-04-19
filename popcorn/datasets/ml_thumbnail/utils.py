# URLs for accessing Popcorn dataset
BASE_URL = "https://huggingface.co/datasets/alitourani/movielens-25m-thumb"
RAW_DATA_URL = f"{BASE_URL}/resolve/main/thumbnails_ml25m_part{{part_id}}.zip"
EMBEDDINGS_URL = (
    f"{BASE_URL}/raw/main/embedding_{{variant}}/thumbnails_ml25m_part{{part_id}}.csv"
)

# Maximum number of parts
MAX_PARTS = 13

# Supported variants
SUPPORTED_VARIANTS = ["raw_frame", "clip", "dino-v2"]


# Some checking functions
def isValidVariant(variant: str) -> bool:
    isValid = variant in SUPPORTED_VARIANTS
    if not isValid:
        print(f"- [Error] Invalid CNN model '{variant}'. Choose from {SUPPORTED_VARIANTS}.")
    return isValid


def isValidPart(part_id: int) -> bool:
    isValid = 1 <= part_id <= MAX_PARTS
    if not isValid:
        print(f"- [Error] Invalid part ID '{part_id}'. Choose from 1 to {MAX_PARTS}.")
    return isValid
