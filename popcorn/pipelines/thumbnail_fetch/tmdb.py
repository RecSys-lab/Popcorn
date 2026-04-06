import requests
from typing import Optional
from popcorn.pipelines.thumbnail_fetch.utils import TMDB_FIND_URL, HEADERS


def fetchPosterPath(
    imdbId: str, tmdbApiKey: str, session: requests.Session
) -> Optional[str]:
    """
    Use the TMDB API to find and fetch the poster path for a movie.

    Parameters
    ----------
    imdbId: str
        The IMDb ID of the movie (e.g. 'tt0111161').
    tmdbApiKey: str
        Your TMDB API key for authentication.
    session: requests.Session
        The requests session to use for making API calls.

    Returns
    -------
    Optional[str]
        The poster path (e.g. '/abc123.jpg') if found, else None.
    """
    # Variables
    url = TMDB_FIND_URL.format(imdb_id=imdbId)
    params = {
        "api_key": tmdbApiKey,
        "external_source": "imdb_id",
    }
    # Make the API request
    try:
        resp = session.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print("-- [Error] TMDB API request failed for %s: %s", imdbId, exc)
        return None

    # Process the results (can be movies, TV shows, or episodes)
    for result_key in ("movie_results", "tv_results", "tv_episode_results"):
        results = data.get(result_key, [])
        if results:
            poster = results[0].get("poster_path")
            if poster:
                # print("-- Found poster_path '%s' via %s", poster, result_key)
                return poster

    print("-- [Warning] No TMDB results found for IMDb ID: %s", imdbId)
    return None


def buildPosterUrl(posterPath: str, size: str = "w500") -> str:
    """
    Build a full TMDB image URL from a posterPath.
    Common sizes: [w92, w154, w185, w342, w500, w780, original]

    Parameters
    ----------
    posterPath: str
        The path to the poster image (e.g. '/abc123.jpg').
    size: str
        The desired size of the image (default 'w500').

    Returns
    -------
    str
        The full URL to the poster image.
    """
    base = f"https://image.tmdb.org/t/p/{size}"
    return f"{base}{posterPath}"
