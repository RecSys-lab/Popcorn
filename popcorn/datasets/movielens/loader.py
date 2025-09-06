import os
import numpy as np
import pandas as pd
from popcorn.datasets.utils import applyKcore
from popcorn.datasets.movielens.downloader import downloadMovieLens
from popcorn.datasets.movielens.helpers import allGenres, itemCols_100k, itemCols, userCols, ratingCols


def loadMovieLens(config: dict):
    """
    Load and prepare the MovieLens dataset based on the provided configuration.

    Parameters
    ----------
    config: dict
        The configuration dictionary containing experiment and dataset settings.

    Returns
    -------
    itemsDF: pd.DataFrame
        The DataFrame containing item (movie) data.
    usersDF: pd.DataFrame
        The DataFrame containing user data.
    ratingsDF: pd.DataFrame
        The DataFrame containing user-item interaction (ratings) data.
    """
    # Variables
    itemsDF = pd.DataFrame()
    usersDF = pd.DataFrame()
    ratingsDF = pd.DataFrame()
    ROOT_PATH = config["general"]["root_path"]
    VERSION = config["datasets"]["unimodal"]["movielens"]["version"]
    DOWNLOAD_PATH = config["datasets"]["unimodal"]["movielens"]["download_path"]
    # Download the dataset (if needed)
    downloadPath = os.path.join(ROOT_PATH, DOWNLOAD_PATH)
    isDownloadSuccessful = downloadMovieLens(VERSION, downloadPath)
    if not isDownloadSuccessful:
        print(f"- Error in loading the 'MovieLens-{VERSION}' dataset! Exiting ...")
        return None, None, None
    # Load the dataset
    datasetRoot = os.path.join(downloadPath, f"ml-{VERSION}", f"ml-{VERSION}")
    datasetRoot = os.path.normpath(datasetRoot)
    print(f"\n- Loading 'MovieLens-{VERSION}' data from '{datasetRoot}' ...")
    if VERSION == "100k":
        filePathUser = os.path.join(datasetRoot, "u.user")
        filePathItem = os.path.join(datasetRoot, "u.item")
        filePathRating = os.path.join(datasetRoot, "u.data")
        delimI, delimU, delimR, eng = "|", "|", "\t", None
    elif VERSION == "1m":
        filePathUser = os.path.join(datasetRoot, "users.dat")
        filePathItem = os.path.join(datasetRoot, "movies.dat")
        filePathRating = os.path.join(datasetRoot, "ratings.dat")
        delimI, eng = "::", "python"
        delimU = delimR = delimI
    elif VERSION == "25m":
        filePathItem = os.path.join(datasetRoot, "movies.csv")
        filePathRating = os.path.join(datasetRoot, "ratings.csv")
        delimI, eng = ",", None
        delimU = delimR = delimI
    try:
        # Read items file
        itemsDF = pd.read_csv(
            filePathItem,
            sep=delimI,
            engine=eng,
            header=None,
            encoding="latin-1",
            skiprows=1 if VERSION == "25m" else 0,
            low_memory=False if eng != "python" else True,
            names=itemCols_100k if VERSION == "100k" else itemCols,
        )
        if VERSION == "100k":
            # Since in MovieLens 100k, genres are in separate columns, we need to combine them into a list
            itemsDF["genres"] = itemsDF[allGenres].apply(
                lambda row: [g for g in allGenres if row[g] == 1], axis=1
            )
            # Now, keep only the relevant columns
            itemsDF = itemsDF[["item_id", "title", "genres"]]
        else:
            itemsDF["genres"] = itemsDF["genres"].map(
                lambda s: s.split("|") if isinstance(s, str) else []
            )
        print(f"- Items (movies) have been loaded. Number of rows: {len(itemsDF):,}")
        # Read users file
        if VERSION == "25m":
            print("- [Note] MovieLens-25M does not provide user metadata! Skipping user data loading ...")
        else:
            usersDF = pd.read_csv(
                filePathUser,
                sep=delimU,
                engine=eng,
                header=None,
                names=userCols,
                encoding="latin-1",
                low_memory=False if eng != "python" else True,
            )
            print(f"- Users have been loaded. Number of rows: {len(usersDF):,}")
        # Read ratings file
        ratingsDF = pd.read_csv(
            filePathRating,
            sep=delimR,
            engine=eng,
            header=None,
            names=ratingCols,
            skiprows=1 if VERSION == "25m" else 0,
            low_memory=False if eng != "python" else True,
        )
        print(f"- Ratings have been loaded. Number of rows: {len(ratingsDF):,}")
        # Preparations
        itemsDF["item_id"] = itemsDF["item_id"].astype(str)
    except Exception as e:
        print(f"- [Error] An error occurred while loading the dataset files: {e}")
        return None, None, None
    # Return
    return itemsDF, usersDF, ratingsDF


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
