def loadGenres(download_path_prefix: str, DATASET: str) -> pd.DataFrame:
    """
    Load genres from the MovieLens dataset.

    Parameters
    ----------
    download_path_prefix : str
        The prefix path where the dataset files are downloaded.
    DATASET : str
        The version of the MovieLens dataset ('100k' or '1m').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their associated genres.
    """
    if DATASET == "100k":
        # Variables
        genre_cols = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        cols = [
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
        ] + genre_cols
        dest_data = os.path.join(download_path_prefix, "u.item")
        # Read the movies file
        movies = pd.read_csv(
            dest_data, sep="|", header=None, names=cols, encoding="latin-1"
        )
        movies["genres"] = movies[genre_cols].apply(
            lambda row: [g for g in genre_cols if row[g] == 1], axis=1
        )
        movies["item_id"] = movies["item_id"].astype(str)
    else:
        # Variables
        dest_folder = os.path.join(download_path_prefix, "ml-1m")
        path = os.path.join(dest_folder, "ml-1m/movies.dat")
        if not os.path.exists(path):
            path = os.path.join(dest_folder, "ml-1m/ml-1m/movies.dat")
        # Read the movies file
        movies = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )
        movies["item_id"] = movies["item_id"].astype(str)
        movies["genres"] = movies["genres"].map(
            lambda s: s.split("|") if isinstance(s, str) else []
        )
    return movies[["item_id", "genres"]]