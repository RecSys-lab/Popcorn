import pandas as pd
from popcorn.utils import loadJsonFromUrl
from popcorn.datasets.popcorn.utils import METADATA_URL
from popcorn.datasets.mmtf14k.utils import VIS_FUSED_URL, VIS_FUSED_FILE_MAP


def getMovieList(dataset: str, configs: dict) -> list[dict]:
    """
    Gets the list of movies from the MovieLens dataset

    Returns
    -------
    movies: list[dict]
        The list of movies from the MovieLens dataset
    """
    movieList = []
    print(f"- Getting the list of movies for the '{dataset}' dataset ...")

    if dataset == "mmtf":
        # Get the address
        addr = VIS_FUSED_URL + VIS_FUSED_FILE_MAP["avf"]
        data = pd.read_csv(addr, low_memory=False)
        # Pick only 'itemId' and 'title'
        dff = data[["itemId", "title"]]
        # Iterate over dff and add to movieList
        for index, row in dff.iterrows():
            movieList.append({"id": row["itemId"], "title": row["title"]})
    elif dataset == "popcorn":
        # Load Popcorn Dataset metadata
        datasetName = configs["datasets"]["multimodal"]["popcorn"]["name"]
        print(
            f"- Loading the '{datasetName}' dataset metadata from '{METADATA_URL}' ..."
        )
        jsonData = loadJsonFromUrl(METADATA_URL)
        if jsonData is None:
            print("- Error in loading the Popcorn dataset metadata! Exiting ...")
            return
        if jsonData:
            movieList = [
                {"id": movie["id"], "title": f'{movie["title"]} ({movie["year"]})'}
                for movie in jsonData
            ]
    else:
        movieList = [
            {"id": "tt0111161", "title": "The Shawshank Redemption (1994)"},
            {"id": "tt0068646", "title": "The Godfather (1972)"},
            {"id": "tt0468569", "title": "The Dark Knight (2008)"},
        ]
    # Print
    if movieList:
        print(f"- Loaded {len(movieList)} movies, such as {movieList[0]}")
    else:
        print("- [Warn] No movies loaded!")
    return movieList
