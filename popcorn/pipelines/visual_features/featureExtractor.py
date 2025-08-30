import os
from popcorn.pipelines.visual_features.models.vgg19 import initModelVgg19
from popcorn.pipelines.visual_features.models.inception3 import initModelInception3
from popcorn.pipelines.visual_features.utils import initFeaturesFolder, modelRunner

def extractMovieFeatures(configs: dict, movieFramesPaths: list):
    """
    Extracts features from the given set of extracted movie frames

    Parameters
    ----------
    configs :dict
        The configurations dictionary
    movieFramesPaths :list
        The list of movie frames paths
    """
    print("Extracting visual features from the given set of extracted movie frames ...")
    # Check the feature extraction model
    model = None
    modelName = configs['feature_extractor_model']
    if modelName == 'incp3':
        model = initModelInception3()
    elif modelName == 'vgg19':
        model = initModelVgg19()
    else:
        print(f"Feature extraction model '{modelName}' is not supported! Exiting ...")
        return
    # Check the model
    if not model:
        print(f"Error while initializing the feature extraction model '{modelName}'! Exiting ...")
        return
    # Iterate on all frame folders in the given root directory
    for framesFolder in movieFramesPaths:
        # Preparing the output frames directory
        framesFolder = os.path.normpath(framesFolder)
        print(f"- Extracting features from the frames in '{framesFolder}' ...")
        outputDir = initFeaturesFolder(framesFolder, configs['features_path'])
        # Skip if the output directory already exists
        if not outputDir:
            continue
        # Extracting features from the frames
        modelRunner(model, framesFolder, outputDir, configs)
