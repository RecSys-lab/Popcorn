import os
import zipfile
import requests
import numpy as np
import pandas as pd
from popcorn.datasets.utils import applyKcore

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def getMovieLensURL(version: str) -> str:
    """
    Get the download URL for the specified MovieLens dataset version.
    
    Parameters
    ----------
    version: str
        The version of the MovieLens dataset ('100k', '1m', or '25m')
    
    Returns
    -------
    url: str
        The download URL for the specified version
    """
    if version == "100k":
        return ML100K_URL
    elif version == "1m":
        return ML1M_URL
    elif version == "25m":
        return ML25M_URL
    else:
        print(f"- Error: Invalid MovieLens version '{version}'. Choose from '100k', '1m', or '25m'.")
        return None

def downloadMovieLens(version: str, downlpadPath: str):
    """
    Downloads the MovieLens 25M dataset

    Parameters
    ----------
    version: str
        The version of the MovieLens dataset to download ('100k', '1m', or '25m')
    downlpadPath: str
        The path to download the dataset to
    
    Returns
    -------
    status: bool
        The status of the download
    """
    print(f"\n- Downloading the MovieLens-{version} dataset ...")
    # If the donwload path does not exist, create it and download the dataset
    downlpadPath = os.path.normpath(downlpadPath)
    if not os.path.exists(downlpadPath):
        print(f"- Creating the download path '{downlpadPath}' ...")
        os.makedirs(downlpadPath)
    else:
        print(f"- The download path '{downlpadPath}' already exists! Skipping the download ...")
        return True
    # Fetch the dataset
    try:
        # Get the download URL
        url = getMovieLensURL(version)
        if url is None:
            return False
        # Download the dataset
        print(f"- Fetching data from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        # Save the downloaded file
        datasetZip = os.path.join(downlpadPath, f'ml-{version}.zip')
        with open(datasetZip, 'wb') as file:
            file.write(response.content)
        # Inform the user
        print("- Download completed and the dataset is saved as a 'zip' file!")
        # Extract the dataset
        print(f"- Now, extracting the dataset files inside {downlpadPath} ...")
        with zipfile.ZipFile(datasetZip, 'r') as zipRef:
            zipRef.extractall(downlpadPath)
        print(f"- Dataset extracted to '{downlpadPath}' successfully!")
        # Remove the zip file after extraction
        print(f"- Removing the zip file {datasetZip} ...")
        os.remove(datasetZip)
        print("- Zip file removed successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"- Error fetching data from {url}: {e}\n")
        return False

# def downloadMovieLens(url: str, downlpadPath: str, VERBOSE: bool = True):
#     """
#     Download a MovieLens dataset file if it does not exist.

#     Parameters
#     ----------
#     url : str
#         The URL to download the dataset from.
#     dest : str
#         The destination file path where the dataset will be saved.
#     VERBOSE : bool, optional
#         If True, print download messages. Default is True.
#     """
#     if not os.path.exists(dest):
#         # Go to a 'data' directory if it exists, otherwise create it
#         if VERBOSE:
#             print(f"⏬ Download {dest}")
#         open(dest, "wb").write(requests.get(url).content)
#     else:
#         if VERBOSE:
#             print(f"✅ '{dest}' already exists, skipping download.")

# def prepareML(config: dict):
#     # Variables
#     SEED = config["experiment"]["seed"]
#     K_CORE = config["experiment"]["k_core"]
#     VERBOSE = config["experiment"]["verbose"]
#     ROOT_PATH = config["general"]["root_path"]
#     SPLIT_MODE = config["experiment"]["split"]["mode"]
#     TEST_RATIO = config["experiment"]["split"]["test_ratio"]
#     DATASET = config["datasets"]['unimodal_dataset']['movielens']["ml_version"]
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