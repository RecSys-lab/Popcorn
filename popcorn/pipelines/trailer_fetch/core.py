import yt_dlp as ytdlp


def getTrailerYoutubeLink(name: str, year: int) -> str:
    """
    Gets the YouTube link for the trailer of a movie

    Parameters
    ----------
    name: str
        The name of the movie
    year: int
        The year of the movie

    Returns
    -------
    link: str
        The YouTube link for the trailer of the movie
    """
    # Variables
    link = None
    queryTerm = f"{name} trailer ({year})"
    print(f"-- Searching for '{queryTerm}' on YouTube ...")
    # Options for yt-dlp
    options = {
        "quiet": True,
        "skip_download": True,
        "default_search": "ytsearch1",
    }
    # Search
    try:
        with ytdlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(queryTerm, download=False)
            if "entries" in info and len(info["entries"]) > 0:
                video = info["entries"][0]
                print(f"-- Found '{video['title']}' in the link '{video['webpage_url']}'! Picking this one ...")
                link = video["webpage_url"]
                return link
            else:
                print(f"- [Warn] No results found for '{queryTerm}'")
                return None
    except Exception as e:
        print(f"- [Warn] Exception in searching YouTube for '{queryTerm}': {e}")
        return None


def generateTrailersYTLinks(movies: list):
    """
    Generates the download links for the trailers of the movies

    Parameters
    ----------
    movies: list
        The list of movies to generate their trailer links.

    Returns
    -------
    trailerLinks: list
        The list of download links for the trailers of the movies
    """
    # Check the validity of the movies list
    if not movies:
        print("- [Warn] The movies list is empty. Cannot generate trailer links ...")
        return []
    # Generate the YouTube links
    print(f"- Generating the YouTube links for the given {len(movies)} movies ...")
    # Prepare the list of download links
    trailerLinks = []
    # Iterate through the movies list
    for movie in movies:
        # Get the YouTube link for the trailer
        link = getTrailerYoutubeLink(movie["title"], movie["year"])
        # Append the link to the list (if found)
        if link:
            trailerLinks.append({"id": movie["id"], "title": movie["title"], "link": link})
        else:
            print(f"- [Warn] Skipping the trailer '{movie['title']}' ({movie['year']}) ...")
    # Return the list of download links
    return trailerLinks


def downloadTrailers(configs: dict, movies: list):
    """
    Downloads the trailers of the given movie list.

    Parameters
    ----------
    configs: dict
        The configurations dictionary of the framework.
    movies: list
        The list of movies to download their trailers.
    """
    # Check the validity of the movies list
    if not movies:
        print("- [Warn] The movies list is empty. Cannot download trailers ...")
        return
    # Prepare the list of trailers YouTube links
    trailers  = generateTrailersYTLinks(movies)
    if not trailers:
        print("- [Warn] No valid trailer links found. Cannot download trailers ...")
        return
    # Download the trailers
    downloadPath = configs["pipelines"]["trailer_fetch"]["download_path"]
    print(f"- Downloading the fetched {len(trailers)} trailers in '{downloadPath}' ...")
    # Iterate through the trailers YouTube links
    for trailer in trailers:
        print(
            f"- Downloading the trailer of '{trailer['title']}' from {trailer['link']} ..."
        )
        # Variables
        options = {
            "outtmpl": f"{configs.get('download_path', '.')}/{trailer['id']}.%(ext)s",
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",  # ensure mp4 output
            "quiet": False,
        }
        try:
            with ytdlp.YoutubeDL(options) as ydl:
                ydl.download([trailer["link"]])
            print(f"- Downloaded '{trailer['title']}' successfully!\n")
        except Exception as e:
            print(f"- [Error] Failed to download '{trailer['title']}': {e}")
