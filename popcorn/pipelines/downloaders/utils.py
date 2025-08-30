#!/usr/bin/env python3
import os
import requests

def filterMovieList(jsonData: list):
    """
    Filters the movie list from the given JSON data to have only the `id`, `title`, and `year` fields
    
    Parameters
    ----------
    jsonData: list
        The JSON data containing the movies
    """
    # Check json file
    if jsonData is None:
        return None
    # Initialize an empty list
    filteredMovies = []
    # Iterate through the JSON data
    for item in jsonData:
        filteredMovies.append({'id': item['id'],
                                'title': item['title'],
                                'year': item['year']
                                })
    # Prepare a log message
    print(f"- Prepared {len(filteredMovies)} movies data from the JSON data to query YouTube ...\n")
    # Return
    return filteredMovies

def videoFileDownloader(url: str, downloadPath: str, fileName: str):
    """
    Downloads a video file from the given URL

    Parameters
    ----------
    url: str
        The URL of the video file
    downloadPath: str
        The path to save the downloaded video file
    fileName: str
        The name of the downloaded video file
    """
    # Check the URL
    if url is None:
        return
    # Download the video file
    print(f"- Downloading the video file from '{url}' ...")
    # Download the video file
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        # Create download address
        downloadPath = os.path.join(downloadPath, fileName)
        # Save the downloaded file
        with open(downloadPath, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        print(f"Error downloading the video file from '{url}'! {e}")
    print(f"- Video file downloaded successfully to '{downloadPath}'!")