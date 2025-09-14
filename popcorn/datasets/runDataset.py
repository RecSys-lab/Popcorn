#!/usr/bin/env python3

from popcorn.utils import loadDataFromCSV, loadJsonFromUrl
from popcorn.datasets.popcorn.visualizer_metadata import visualizeGenresDictionary
from popcorn.datasets.popcorn.helper_visualfeats import packetAddressGenerator, fetchAllPackets
from popcorn.datasets.popcorn.helper_metadata import countNumberOfMovies, fetchRandomMovie, fetchMovieById
from popcorn.datasets.movielens.helper_movies import fetchAllUniqueGenres, fetchMoviesByGenre as fetchMoviesByGenreMovielens
from popcorn.datasets.popcorn.helper_metadata import classifyYearsByCount, fetchMoviesByGenre, classifyMoviesByGenre, calculateAverageGenrePerMovie
from popcorn.datasets.popcorn.helper_visualfeats_agg import fetchAggregatedFeatures, generatedAggFeatureAddresses, loadAggregatedFeaturesIntoDataFrame

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


