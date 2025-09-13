#!/usr/bin/env python3

from popcorn.utils import loadJsonFromUrl

def packetAddressGenerator(datasetUrl: str, gFeature: str, gModel: str, gMovieId, gPacketId):
  """
  Generates the address of a packet file based on the given parameters.

  Parameters:
      datasetUrl (str): The base URL of the dataset.
      gFeature (str): The feature type (e.g., "audio", "visual").
      gModel (str): The model used for feature extraction.
      gMovieId (int): The ID of the movie.
      gPacketId (int): The ID of the packet.

  Returns:
      packetAddress (str): The URL address of the
  """
  # Standardize variables
  gMovieId = f"{int(gMovieId):010d}"
  gPacketId = str(gPacketId).zfill(4)
  # Create address
  packetAddress = datasetUrl + f"{gFeature}/{gModel}/{gMovieId}/packet" + str(gPacketId).zfill(4) + ".json"
  return packetAddress

def fetchAllPackets(datasetUrl: str, gFeature: str, gModel: str, gMovieId):
  """
  Fetches all packets of a movie from the dataset.

  Parameters:
      datasetUrl (str): The base URL of the dataset.
      gFeature (str): The feature type (e.g., "audio", "visual").
      gModel (str): The model used for feature extraction.
      gMovieId (int): The ID of the movie.

  Returns:
      moviePackets (list): A list of all packets of the movie.  
  """
  # Variables
  counter = 0
  moviePackets = []
  # Loop over all possible files
  while True:
    counter += 1
    # Generate packet address
    packetAddress = packetAddressGenerator(datasetUrl, gFeature, gModel, gMovieId, counter)
    print(f'Generated packet address: {packetAddress}')
    # Fetch JSON data
    jsonData = loadJsonFromUrl(packetAddress)
    if jsonData:
      print(f'Fetched JSON from the address ...')
      moviePackets += jsonData
    else:
      print(f'No JSON data found at the address ...')
      break
  # Return
  return moviePackets