#!/usr/bin/env python3

from movifex.utils import readConfigs
from movifex.pipelines.downloaders.utils import videoFileDownloader
from movifex.multimodal.fused.fuse_audio_textual import fuseTextualWithMMTFAudio
from movifex.multimodal.fused.overlap_checker import runVisualTextualDatasetsOverlapChecker
from movifex.datasets.runDataset import testMoViFexMetadata, testMoViFexEmbeddings, testMovieLens25M
from movifex.runCore import runShotDetectionFromFrames, runShotDetectionFromFeatures, runAggFeatures
from movifex.multimodal.fused.fuse_visual_textual import fuseTextualWithMMTFVisual, fuseTextualWithSceneSense
from movifex.runCore import runTrailerDownloader, runMoviesFrameExtractor, runMoviesFramesFeatureExtractor

def main():
    print("Welcome! Starting 'MoViFex'!\n")
    # Read the configuration file
    configs = {}
    configs = readConfigs("movifex/config/config.yml")
    # If properly read, print the configurations
    if not configs:
        print("Error loading configuration parameters!")
        return
    print("Configuration file loaded successfully!\n")
    # Get common configurations
    cfgGeneral = configs['config']['general']
    cfgDatasets = configs['config']['datasets']
    cfgRecSys = configs['config']['multimodal']
    cfgPipeline = configs['config']['pipelines']
    # Check which mode to run
    mode = cfgGeneral['mode']
    if (mode == 'pipeline'):
        # Get the selected sub-mode
        subMode = cfgGeneral['sub_mode_pipeline']
        if (subMode == 'dl_trailers'):
            # Run the trailer downloader pipeline
            runTrailerDownloader(cfgPipeline['movie_trailers'], cfgDatasets['visual_dataset']['movifex'])
        elif (subMode == 'frame_extractor'):
            # Download a sample video file (optional)
            # downloadPath = cfgPipeline['movie_frames']['movies_path']
            # videoUrl = 'https://cdn.pixabay.com/video/2024/02/28/202368-918049003_large.mp4'
            # videoFileDownloader(videoUrl, downloadPath, fileName='test.mp4')
            # Run the movies frame extractor pipeline
            runMoviesFrameExtractor(cfgPipeline['movie_frames'])
        elif (subMode == 'feat_extractor'):
            # Run the movies features extractor pipeline
            runMoviesFramesFeatureExtractor(cfgPipeline['movie_frames_visual_features'])
        elif (subMode == 'shot_from_feat'):
            # Run the shot detection pipeline from the extracted features
            runShotDetectionFromFeatures(cfgPipeline['movie_shots']['variants']['from_features'])
        elif (subMode == 'shot_from_frame'):
            # Run the shot detection pipeline from the extracted frames
            runShotDetectionFromFrames(cfgPipeline['movie_shots']['variants']['from_frames'])
        elif (subMode == 'agg_features'):
            # Run the feature aggregation pipeline
            runAggFeatures(cfgPipeline['feature_aggregation'])
        else:
            print(f"Unsupported sub-mode '{subMode}' selected! Exiting ...")
    elif (mode == 'ds'):
        # Get the selected sub-mode
        subMode = cfgGeneral['sub_mode_ds']
        if (subMode == 'movifex_meta'):
            testMoViFexMetadata(cfgDatasets['visual_dataset']['movifex'])
        elif (subMode == 'movifex_visual'):
            testMoViFexEmbeddings(cfgDatasets['visual_dataset']['movifex'])
        elif (subMode == 'movielens_25m'):
            testMovieLens25M(cfgDatasets['text_dataset'])
        else:
            print(f"Unsupported sub-mode '{subMode}' selected! Exiting ...")
    elif (mode == 'recsys'):
        # Get the selected sub-mode
        subMode = cfgGeneral['sub_mode_recsys']
        if (subMode == 'overlap_checker'):
            # Check the overlap between the SceneSense, MMTF, and the LLM-enriched dataset for recommendation
            runVisualTextualDatasetsOverlapChecker(cfgRecSys, cfgDatasets)
        elif (subMode == 'visual_text_fusion'):
            # Fusion of the visual (MMTF) and textual features for recommendation
            # fuseTextualWithMMTFVisual(cfgRecSys, cfgDatasets)
            # Fusion of the visual (SceneSense) and textual features for recommendation
            fuseTextualWithSceneSense(cfgRecSys, cfgDatasets)
        elif (subMode == 'audio_text_fusion'):
            # Fusion of the audio (MMTF) and textual features for recommendation
            fuseTextualWithMMTFAudio(cfgRecSys, cfgDatasets)
    else:
        print(f"Unsupported mode '{mode}' selected! Exiting ...")
    # Finish the program
    print("\nStopping 'MoViFex'!")

if __name__ == "__main__":
    main()