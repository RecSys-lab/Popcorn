import os
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from popcorn.pipelines.thumbnail_fetch.tmdb import fetchPosterPath, buildPosterUrl
from popcorn.pipelines.thumbnail_fetch.utils import (
    movielensToIMDB,
    SUPPORTED_POSTER_SIZES,
    DEFAULT_OUTPUT_DIR_PREFIX,
    SUPPORTED_DATASET_VARIANTS,
)


def downloadImage(url: str, destPath: Path, session: requests.Session) -> bool:
    """
    Download an image to a specified destination path using the provided requests session.

    Parameters
    ----------
    url: str
        The URL of the image to download.
    destPath: Path
        The destination file path to save the downloaded image.
    session: requests.Session
        The requests session to use for making the download request.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    try:
        resp = session.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        destPath.parent.mkdir(parents=True, exist_ok=True)
        with open(destPath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        return True
    except requests.RequestException as exc:
        print("-- [Error] Failed to download image from %s: %s", url, exc)
        return False


def downloadThumbnail(
    movie: dict,
    dataset: str,
    linksDF: pd.DataFrame,
    downloadPath: str,
    session: Optional[requests.Session],
    posterSize: str,
    delay: float,
    configs: dict,
) -> dict:
    """
    Fetch and download the TMDB poster for a single movie.

    Parameters
    ----------
    movie: dict
        A dictionary containing movie information, including 'id' and 'title'.
    dataset: str
        The dataset variant being used (e.g. 'dummy', 'mmtf', 'popcorn').
    downloadPath: str
        The base directory path where the poster should be saved.
    session: Optional[requests.Session]
        An optional requests session to use for API calls. If None, a new session will be created and closed within this function.
    posterSize: str
        The desired size of the poster image (default 'w500').
    delay: float
        Delay in seconds to wait after processing this movie before proceeding to the next one (default 0.25).

    Returns
    -------
    dict
        A dictionary containing the download result, including 'id', 'title', 'url', 'path', and 'success' keys.
    """
    # Variables
    apiKey = configs["pipelines"]["thumbnail_fetch"]["tmdb_api_key"]
    # Session management
    ownSession = session is None
    if ownSession:
        session = requests.Session()
    # Get IMDb ID and title from the movie dict
    imdbId = (
        movie["id"]
        if dataset == "dummy"
        else movielensToIMDB(int(movie["id"]), linksDF)
    )
    title = movie["title"]
    result = {
        "id": imdbId,
        "title": title,
        "url": None,
        "path": None,
        "success": False,
    }
    # Process the movie
    try:
        print(f"-- Processing {title} ({imdbId}) ...")
        posterPath = fetchPosterPath(imdbId, apiKey, session)
        if not posterPath:
            return result
        # Build the full image URL and destination path
        imgUrl = buildPosterUrl(posterPath, size=posterSize)
        ext = Path(posterPath).suffix or ".jpg"
        filename = str(movie["id"]) + ext  # 'safeFilename(title) + ext'
        dest = Path(downloadPath) / filename
        # Download the image
        if downloadImage(imgUrl, dest, session):
            result.update(url=imgUrl, path=str(dest), success=True)
            print(f"-- Saved → {dest}")
        time.sleep(delay)
    finally:
        if ownSession:
            session.close()

    return result


def downloadThumbnails(
    movies: list[dict], linksDF: pd.DataFrame, configs: dict, delay: float = 0.25
) -> list[dict]:
    """
    Fetch and download TMDB posters for a list of movies.

    Parameters
    ----------
    movies: list[dict]
        List of dicts, like [{"id": "tt0111161", "title": "The Shawshank Redemption (1994)"}, ...]
    configs: dict
        Framework configuration options.
    delay: float
        Delay in seconds between API calls (TMDB rate limit: 50 req/s).

    Returns
    -------
    list[dict]
        List of result dicts with keys: id, title, url, path, success.
    """
    # Variables
    results = []
    API_KEY = configs["pipelines"]["thumbnail_fetch"]["tmdb_api_key"]
    DATASET = configs["pipelines"]["thumbnail_fetch"]["dataset_variant"]
    POSTER_SIZE = configs["pipelines"]["thumbnail_fetch"]["poster_size"]
    DOWNLOAD_PATH = configs["pipelines"]["thumbnail_fetch"]["download_path"]
    print(
        f"- Starting thumbnail download for {len(movies)} movies using dataset variant '{DATASET}' ..."
    )
    # Check arguments
    if not API_KEY or API_KEY == "YOUR_KEY_HERE":
        print(f"-- [Error] TMDB API key is missing in the configuration. Exiting ...")
        return results
    if len(movies) < 1:
        print(f"-- [Error] The list of movies is empty. No thumbnails to download.")
        return results
    if delay < 0 or delay > 5.0:
        print(f"-- [Warn] Changing delay value to 0.25 ...")
        delay = 0.25
    if POSTER_SIZE not in SUPPORTED_POSTER_SIZES:
        print(
            f"-- [Warn] Invalid poster size '{POSTER_SIZE}'. Defaulting to 'w500' ..."
        )
        POSTER_SIZE = "w500"
    if DATASET not in SUPPORTED_DATASET_VARIANTS:
        print(
            f"-- [Warn] Unsupported dataset variant '{DATASET}'. Defaulting to 'dummy' ..."
        )
        DATASET = "dummy"
    # If the download path does not exist, create it and download the dataset
    downloadPath = os.path.normpath(DOWNLOAD_PATH)
    downloadPath = os.path.join(downloadPath, f"{DEFAULT_OUTPUT_DIR_PREFIX}{DATASET}")
    if not os.path.exists(downloadPath):
        print(f"- Creating the download path '{downloadPath}' ...")
        os.makedirs(downloadPath)
    else:
        print(
            f"- The download path '{downloadPath}' already exists! Skipping the download ..."
        )
        return results
    # Download thumbnails
    with requests.Session() as session:
        for movie in movies:
            result = downloadThumbnail(
                movie,
                DATASET,
                linksDF,
                downloadPath,
                session,
                POSTER_SIZE,
                delay,
                configs,
            )
            results.append(result)
    # Summary
    print("\nSummary:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"- {status} {r['title']}")
        if r["path"]:
            print(f"      → {r['path']}")
    return results

