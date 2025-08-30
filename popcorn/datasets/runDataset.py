#!/usr/bin/env python3

import os
from popcorn.utils import loadDataFromCSV, loadJsonFromUrl
from popcorn.datasets.utils import printTextualDatasetStats
from popcorn.datasets.movielens.downloader import downloadMovielens25m
from popcorn.datasets.movifex.visualizer_metadata import visualizeGenresDictionary
from popcorn.datasets.movielens.helper_ratings import mergeMainGenreMoviesDFWithRatingsDF
from popcorn.datasets.movifex.helper_visualfeats import packetAddressGenerator, fetchAllPackets
from popcorn.datasets.movifex.helper_metadata import countNumberOfMovies, fetchRandomMovie, fetchMovieById
from popcorn.datasets.movielens.helper_movies import fetchAllUniqueGenres, fetchMoviesByGenre as fetchMoviesByGenreMovielens
from popcorn.datasets.movifex.helper_metadata import classifyYearsByCount, fetchMoviesByGenre, classifyMoviesByGenre, calculateAverageGenrePerMovie
from popcorn.datasets.movielens.helper_movies import augmentMoviesDFWithBinarizedGenres, binarizeMovieGenres, filterMoviesWithMainGenres, mainGenres
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

def testMovieLens25M(configs: dict):
    """
    Runs the text dataset pipeline (MovieLens 25M dataset)

    Parameters
    ----------
    configs :dict
        The configurations dictionary
    """
    print(f"Running the text dataset pipeline '{configs['name']}' ...")
    # Variables
    needDownloaded = configs['need_download']
    datasetPath = os.path.normpath(configs['download_path'])
    # Pre-check whether the dataset is already downloaded
    print(f"- Checking if the dataset needs to be downloaded (selected: {needDownloaded}) ...")
    if needDownloaded:
        print(f"- The dataset needs to be downloaded! It will be downloaded in '{datasetPath}' ...")
        isDownloadSuccessful = downloadMovielens25m(configs['url'], datasetPath)
        if not isDownloadSuccessful:
            return
        # Go inside the downloaded folder
        datasetPath = os.path.join(datasetPath, "ml-25m")
        print(f"- The dataset files are available in '{datasetPath}'!")
    else:
        # Else, the dataset is already downloaded (reading from the root folder)
        print(f"- The dataset is already downloaded! Trying to load it from '{datasetPath}' ...")
    # Some test functions
    print(f"\nRunning some functionalities provided in the framework for the '{configs['name']}' dataset ...")
    # Reading movies data
    print(f"- Reading dataset's movies and fetching them into a DataFrame ...")
    moviesDataFrame = loadDataFromCSV(os.path.join(datasetPath, "movies.csv"))
    if moviesDataFrame is None:
        print(f"- The dataset files could not be found! Checking the inner folder ...")
        # Go inside the 'ml-25m' folder
        datasetPath = os.path.join(datasetPath, "ml-25m")
        moviesDataFrame = loadDataFromCSV(os.path.join(datasetPath, "movies.csv"))
        if moviesDataFrame is None:
            print(f"- The dataset files could not be found! Exiting ...")
            return
    # Test#1 - Counting the number of movies
    moviesCount = len(moviesDataFrame)
    print(f"- The dataset contains {moviesCount} movies!")
    # Test#2 - Some samples of the movies
    print(f"- The structure of the movies data is as below:")
    print(moviesDataFrame.head(3))
    # Test#3 - Get all genres from the dataset in a list
    print(f"\n- Fetching all genres from the dataset ...")
    allGenres = fetchAllUniqueGenres(moviesDataFrame)
    print(f"- The dataset contains {len(allGenres)} genres, including: {allGenres}")
    # Test#4 - Get movies by a specific genre
    givenGenre = "Action"
    print(f"\n- Fetching movies by a specific genre (given: {givenGenre}) ...")
    moviesByGenre = fetchMoviesByGenreMovielens(moviesDataFrame, givenGenre)
    print(f"- The dataset contains {len(moviesByGenre)} movies with the genre '{givenGenre}'!")
    # Test#5 - Get movies by the main genres
    print(f"\n- Fetching movies by the main genres {mainGenres} ...")
    mainGenresMoviesDataFrame = filterMoviesWithMainGenres(moviesDataFrame)
    print(f"- The dataset contains {len(mainGenresMoviesDataFrame)} movies with the main genres!")
    print(f"- A sample of the movies with the main genres: \n{mainGenresMoviesDataFrame.head(3)}")
    # Test#6 - Model movies data with binarized genres
    moviesDFBinarizedGenres = binarizeMovieGenres(moviesDataFrame)
    print(f"\n- The movies data with binarized genres is as below: \n{moviesDFBinarizedGenres.head(3)}")
    # Test#7 - Augment the movies data with the binarized genres
    augmentedMoviesDataFrame = augmentMoviesDFWithBinarizedGenres(moviesDataFrame, moviesDFBinarizedGenres)
    print(f"\n- The movie data augmented with binary genres is as below: \n{augmentedMoviesDataFrame.head(3)}")
    # Test#8 - Reading user-driven data
    print(f"\n- Reading dataset's user-driven data and fetching them into a DataFrame ...")
    ratingsDataFrame = loadDataFromCSV(os.path.join(datasetPath, "ratings.csv"))
    if ratingsDataFrame is None:
        return
    # Test#9 - Counting the number of ratings
    ratingsCount = len(ratingsDataFrame)
    print(f"- The dataset contains {ratingsCount} ratings!")
    # Test#10 - Some samples of the movies
    print(f"- The structure of the user-ratings data is as below:\n{ratingsDataFrame.head(3)}")
    # Test#11 - Merging the movies and ratings DataFrames
    print(f"\n- Merging the movies (of the main genres) and ratings DataFrames ...")
    mergedDataFrame = mergeMainGenreMoviesDFWithRatingsDF(augmentedMoviesDataFrame, ratingsDataFrame)
    print(f"- The merged DataFrame has {len(mergedDataFrame)} items, such as:\n{mergedDataFrame.head(3)}")
    # Test#12 - Dataset statistics
    print(f"\n- Checking dataset statistics ...")
    ratingsDataFrameRenamed = ratingsDataFrame.rename(columns={'movieId': 'itemId'})
    printTextualDatasetStats(ratingsDataFrameRenamed)
