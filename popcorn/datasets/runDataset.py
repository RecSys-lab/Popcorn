#!/usr/bin/env python3

from popcorn.utils import loadDataFromCSV, loadJsonFromUrl
from popcorn.datasets.movifex.visualizer_metadata import visualizeGenresDictionary
from popcorn.datasets.movifex.helper_visualfeats import packetAddressGenerator, fetchAllPackets
from popcorn.datasets.movifex.helper_metadata import countNumberOfMovies, fetchRandomMovie, fetchMovieById
from popcorn.datasets.movielens.helper_movies import fetchAllUniqueGenres, fetchMoviesByGenre as fetchMoviesByGenreMovielens
from popcorn.datasets.movifex.helper_metadata import classifyYearsByCount, fetchMoviesByGenre, classifyMoviesByGenre, calculateAverageGenrePerMovie
from popcorn.datasets.movifex.helper_visualfeats_agg import fetchAggregatedFeatures, generatedAggFeatureAddresses, loadAggregatedFeaturesIntoDataFrame

def testMoViFexMetadata(configs: dict):
    """
    Runs the visual dataset pipeline (MoViFex dataset) for metadata processing

    Parameters
    ----------
    configs :dict
        The configurations dictionary
    """
    # Variables
    datasetName = configs['name']
    datasetMetadataUrl = configs['path_metadata']
    print(f"Running the visual dataset functions of '{datasetName}' for metadata (json) processing ...")
    # Pre-check fetch JSON data from the URL
    print(f"- Fetching the dataset metadata from '{datasetMetadataUrl}' ...")
    jsonData = loadJsonFromUrl(datasetMetadataUrl)
    # Test#1 - Movie Counting
    print(f"\n- Counting the number of movies in the dataset ...")
    moviesCount = countNumberOfMovies(jsonData)
    print(f"- The dataset contains '{moviesCount}' movies!")
    # Test#2 - Testing Random Movie Fetcher
    print(f"\n- Fetching a random movie from the dataset ...")
    randomMovie = fetchRandomMovie(jsonData)
    print(f"- The random movie:\n{randomMovie}")
    # Test#3 - Fetching a movie by ID
    givenMovieId = 6
    print(f"\n- Fetching a given movie by ID (given: {givenMovieId}) ...")
    movieById = fetchMovieById(jsonData, givenMovieId)
    print(f"- The fetched movie:\n{movieById}")
    # Test#4 - Fetching movies by genre
    givenGenre = "Romance"
    print(f"\n- Fetching all movie with the given genre (input: {givenGenre}) ...")
    moviesByGenre = fetchMoviesByGenre(jsonData, givenGenre)
    print(f"- Returned variable (list): {moviesByGenre}")
    # Test#5 - Year classification
    print(f"\n- Classifying release dates by count ...")
    yearsCount = classifyYearsByCount(jsonData)
    print(f"- Returned variable (dict): {yearsCount}")
    # Test#6 - Genre classification
    print(f"\n- Classifying movies by genre ...")
    moviesByGenre = classifyMoviesByGenre(jsonData)
    print(f"- Returned variable (dict): {moviesByGenre}")
    # Test#7 - Average genre per movie calculation
    print(f"\n- Calculating the average genre per movie ...")
    averageGenrePerMovie = calculateAverageGenrePerMovie(moviesByGenre, moviesCount)
    print(f"- Returned variable (float): {averageGenrePerMovie}")
    # Test#8 - Visualizations
    print(f"\n- Visualizing the classification results in a bar chart ...")
    visualizeGenresDictionary(moviesByGenre)

def testMoViFexEmbeddings(configs: dict):
    # Variables
    datasetName = configs['name']
    datasetRawFilesUrl = configs['path_raw']
    featureModels = configs['feature_models']
    featureSources = configs['feature_sources']
    aggFeatureSources = configs['agg_feature_sources']
    # Other variables
    givenMovieId = 6
    givenModel = featureModels[0]
    givenFeatureSource = featureSources[2]
    givenAggFeatureSource = aggFeatureSources[0]
    print(f"Running the visual dataset functions of '{datasetName}' embedding processing ...")
    # Pre-check fetch addresses of files
    print(f"\n- Generating a sample packet address file from '{datasetRawFilesUrl}' ...")
    packetAddress = packetAddressGenerator(datasetRawFilesUrl, givenFeatureSource, givenModel, givenMovieId, 1)
    print(f"- Generated address (str): {packetAddress}")
    # Fetch all packets of a movie
    print(f"\n- Fetching all packets of the movie #{givenMovieId}) ...")
    moviePackets = fetchAllPackets(datasetRawFilesUrl, givenFeatureSource, givenModel, givenMovieId)
    print(f"- Number of fetched packets (list): {len(moviePackets)}")
    # Fetch all aggregated features of a given movie
    print(f"\n- Fetching aggregated features of the movie #{givenMovieId} ({givenModel}, {givenAggFeatureSource}) ...")
    aggFeatures = fetchAggregatedFeatures(datasetRawFilesUrl, givenAggFeatureSource, givenModel, givenMovieId)
    print(f"- The fetched aggregated features (list): {aggFeatures}")
    # Fetching aggregated features
    print(f"\n- Generating addresses for aggregated features ...")
    aggFeatureAddresses = generatedAggFeatureAddresses(configs)
    print(f"- Samples of the generated addresses: {aggFeatureAddresses['full_movies_agg']['incp3'][:2]}")
    # Loading a sample aggregated feature into a DataFrame
    print(f"\n- Loading a sample aggregated feature into a DataFrame (Full-Movie, Inception 3.0) ...")
    tmpVisualDFMax, tmpVisualDFMean = loadAggregatedFeaturesIntoDataFrame(aggFeatureAddresses['full_movies_agg']['incp3'])
    print(f"- Loaded {len(tmpVisualDFMax)} records for 'Max' aggregated features! Check the first 3 records:")
    print(f"- The loaded DataFrame (Max): {tmpVisualDFMax.head(3)}")
    print(f"- The loaded DataFrame (Mean): {tmpVisualDFMean.head(3)}")


