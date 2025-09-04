import os
import numpy as np
import pandas as pd
from popcorn.datasets.utils import applyKcore
from popcorn.datasets.movielens.downloader import downloadMovieLens


def loadMovieLens(config: dict):
    """
    Load and prepare the MovieLens dataset based on the provided configuration.

    Parameters
    ----------
    config: dict
        The configuration dictionary containing experiment and dataset settings.

    Returns
    -------
    """
    # Variables
    SEED = config["experiment"]["seed"]
    ROOT_PATH = config["general"]["root_path"]
    VERSION = config["datasets"]["unimodal"]["movielens"]["version"]
    DOWNLOAD_PATH = config["datasets"]["unimodal"]["movielens"]["download_path"]
    # Download the dataset (if needed)
    downloadPath = os.path.join(ROOT_PATH, DOWNLOAD_PATH)
    isDownloadSuccessful = downloadMovieLens(VERSION, downloadPath)
    if not isDownloadSuccessful:
        print(f"- Error in loading the 'MovieLens-{VERSION}' dataset! Exiting ...")
        return
    # Load the dataset
    datasetRoot = os.path.join(downloadPath, f"ml-{VERSION}", f"ml-{VERSION}")
    datasetRoot = os.path.normpath(datasetRoot)
    print(f"\n- Loading 'MovieLens-{VERSION}' data from '{datasetRoot}' ...")
    if VERSION == "100k":
        filePathUser = os.path.join(datasetRoot, "u.user")
        filePathItem = os.path.join(datasetRoot, "u.item")
        filePathRating = os.path.join(datasetRoot, "u.data")
        delim, eng = "\t", None
    elif VERSION == "1m":
        filePathUser = os.path.join(datasetRoot, "users.dat")
        filePathItem = os.path.join(datasetRoot, "movies.dat")
        filePathRating = os.path.join(datasetRoot, "ratings.dat")
        delim, eng = "::", "python"
    elif VERSION == "25m":
        filePathUser = os.path.join(datasetRoot, "tags.csv")
        filePathItem = os.path.join(datasetRoot, "movies.csv")
        filePathRating = os.path.join(datasetRoot, "ratings.csv")
        delim, eng = ",", None
    # Read ratings file
    ratingsDF = pd.read_csv(
        filePathRating,
        sep=delim,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine=eng,
        header=None,
        low_memory=False if eng != "python" else True,
    )
    print(
        f"- Ratings have been loaded. Number of rows: {len(ratingsDF):,}"
    )


# def prepareMovieLens(config: dict):
#     # Variables
#     SEED = config["experiment"]["seed"]
#     K_CORE = config["experiment"]["k_core"]
#     VERBOSE = config["experiment"]["verbose"]
#     ROOT_PATH = config["general"]["root_path"]
#     SPLIT_MODE = config["experiment"]["split"]["mode"]
#     TEST_RATIO = config["experiment"]["split"]["test_ratio"]
#     DATASET = config["datasets"]['unimodal']['movielens']["ml_version"]
#     download_path_prefix = os.path.join(ROOT_PATH, "movifex", "data", "downloaded")
#     # Download the dataset
#     print(f"\nPreparing 'MovieLens {DATASET}' data ...")
#     if DATASET == "100k":
#         # Variables
#         dest_data = os.path.join(download_path_prefix, "u.data")
#         dest_item = os.path.join(download_path_prefix, "u.item")
#         # Download
#         downloadMovieLens(ML100K_URL, dest_data, VERBOSE)
#         downloadMovieLens(ML100K_ITEM, dest_item, VERBOSE)
#         # Separate ratings and items
#         ratings_file, delim, eng = dest_data, "\t", None
#     else:
#         # Variables
#         dest = os.path.join(download_path_prefix, "ml-1m.zip")
#         dest_folder = os.path.join(download_path_prefix, "ml-1m")
#         # Download
#         downloadMovieLens(ML1M_URL, dest, VERBOSE)
#         if not os.path.exists(dest_folder):
#             print(f"⏬ Extracting '{dest}' to '{dest_folder}' ...")
#             zipfile.ZipFile(dest).extractall(dest_folder)
#         ratings_file = (
#             f"{dest_folder}/ml-1m/ratings.dat"
#             if os.path.exists(f"{dest_folder}/ml-1m/ratings.dat")
#             else f"{dest_folder}/ratings.dat"
#         )
#         delim, eng = "::", "python"
#     # Read ratings file
#     ratings = pd.read_csv(
#         ratings_file,
#         sep=delim,
#         names=["user_id", "item_id", "rating", "timestamp"],
#         engine=eng,
#         header=None,
#     )
#     if VERBOSE:
#         print(f"✔ Ratings rows = {len(ratings):,}")
#     # Load genres
#     genres_df = loadGenres(download_path_prefix, DATASET)
#     genre_dict = dict(zip(genres_df.item_id, genres_df.genres))
#     if VERBOSE:
#         print(f"✔ genres loaded items = {len(genres_df):,}")
#     # Apply k-core filtering (if specified)
#     if K_CORE > 0:
#         ratings = applyKcore(ratings, K_CORE)
#         if VERBOSE:
#             print(f"✔ After {K_CORE}-core rows = {len(ratings):,}")
#     # Split the dataset into train and test sets
#     np.random.seed(SEED)
#     if SPLIT_MODE == "random":
#         ratings = ratings.sample(frac=1, random_state=SEED).reset_index(drop=True)
#         sz = int(len(ratings) * TEST_RATIO)
#         train_df, test_df = ratings.iloc[:-sz].copy(), ratings.iloc[-sz:].copy()
#     elif SPLIT_MODE == "temporal":
#         ratings = ratings.sort_values("timestamp")
#         sz = int(len(ratings) * TEST_RATIO)
#         train_df, test_df = ratings.iloc[:-sz].copy(), ratings.iloc[-sz:].copy()
#     else:
#         trs, tes = [], []
#         for uid, grp in ratings.groupby("user_id"):
#             grp = grp.sort_values("timestamp")
#             tes.append(grp.iloc[-1])
#             trs.extend(grp.iloc[:-1].to_dict("records"))
#         train_df, test_df = pd.DataFrame(trs), pd.DataFrame(tes)
#     # Make the train set
#     if VERBOSE:
#         print(f"✔ Split train = {len(train_df):,}  test = {len(test_df):,}")
#     # train_set = Dataset.from_uir(train_df[['user_id','item_id','rating']].values.tolist())
#     # Save the genres DataFrame to a CSV file
#     genres_df_save_path = os.path.join(ROOT_PATH, "outputs", "item_metadata_genres.csv")
#     genres_df.to_csv(genres_df_save_path, index=False)
#     print(f"✔ {genres_df_save_path} saved!")
#     # Return
#     return train_df, test_df, genre_dict
