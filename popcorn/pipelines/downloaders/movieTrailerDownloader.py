from pytube import Search, YouTube

def getTrailerYoutubeLink(name: str, year: int):
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
    # Prepare the search query
    query = Search(f'{name} trailer ({year})')
    # Check the query
    link = None
    if len(query.results) > 0:
        link = query.results[0].watch_url
    # Return the link
    return link

def generateTrailersYouTubeLinks(moviesList: list):
    """
    Generates the download links for the trailers of the movies

    Parameters
    ----------
    moviesList: list
        The list of movies
    
    Returns
    -------
    trailersYouTubeLinks: list
        The list of download links for the trailers of the movies
    """
    print(f"-- Generating the YouTube links for the given movies ...")
    # Prepare the list of download links
    trailersYouTubeLinks = []
    # Iterate through the movies list
    for movie in moviesList:
        trailersYouTubeLinks.append({'id': movie['id'],
                            'title': movie['title'],
                            'link': getTrailerYoutubeLink(movie['title'], movie['year'])})
    # Prepare a log message
    print(f"-- Generated {len(trailersYouTubeLinks)} YouTube links for the movies!\n")
    # Return the list of download links
    return trailersYouTubeLinks

def downloadMovieTrailers(configs: dict, filteredMovies: list):
    """
    Downloads the trailers of the movies

    Parameters
    ----------
    configs: dict
        The configurations of the pipeline
    filteredMovies: list
        The list of filtered movies
    """
    print(f"- Running the {configs['name']} ...")
    # Prepare the list of trailers YouTube links
    trailersYouTubeLinks = generateTrailersYouTubeLinks(filteredMovies)
    print(f"- Downloading the results in {configs['download_path']} ...")
    # Iterate through the trailers YouTube links
    for trailer in trailersYouTubeLinks:
        print(f"- Downloading the trailer of '{trailer['title']}' from {trailer['link']} ...")
        # Check if the link is available
        if trailer['link']:
            # Download the trailer
            YouTube(trailer['link']).streams.first().download(output_path=configs['download_path'], filename=f"{trailer['id']}")
            print(f"- Downloaded the trailer of '{trailer['title']}'!\n")
        else:
            print(f"- Could not download the trailer of '{trailer['title']}'!\n")