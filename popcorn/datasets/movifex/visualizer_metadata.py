#!/usr/bin/env python3

import matplotlib.pyplot as plt

def visualizeGenresDictionary(genresDict: dict):
  """
  Visualize the genres dictionary as a bar chart.

  Parameters:
      genresDict (dict): A dictionary containing genres as keys and their counts as values.
  """
  # Check if the dictionary is empty
  if not genresDict:
      print("Genres dictionary is empty!")
      return
  # Sort genres by count
  sortedGenres = sorted(genresDict.items(), key=lambda x: x[1], reverse=True)
  # Extract the genre names and counts
  genreNames, genreCounts = zip(*sortedGenres)
  # Create the bar chart
  plt.bar(genreNames, genreCounts)
  plt.xlabel("Genres")
  plt.ylabel("Number of Movies")
  plt.title("Movies Classified by Genre")
  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45)
  # Show the plot
  plt.show()