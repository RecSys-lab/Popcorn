import os
from popcorn.pipelines.utils import cnnModels
from popcorn.pipelines.visual_embedding.models.vgg19 import initModelVgg19
from popcorn.pipelines.visual_embedding.models.inception3 import initModelInception3
from popcorn.pipelines.visual_embedding.utils import (
    fetchMovieFramesFolders,
    initFeaturesPath,
    modelRunner,
)


def extractVisualEmbeddings(configs: dict):
    """
    Extracts features from the given set of extracted movie frames

    Parameters
    ----------
    configs: dict
        The configurations dictionary
    """
    # Variables
    model = None
    cnn = configs["pipelines"]["visual_embedding_extractor"]["cnn"]
    pipelineName = configs["pipelines"]["visual_embedding_extractor"]["name"]
    packetSize = configs["pipelines"]["visual_embedding_extractor"]["packet_size"]
    featuresRootPath = configs["pipelines"]["visual_embedding_extractor"][
        "features_path"
    ]
    print(f"- Starting the '{pipelineName}' pipeline ...")
    # Check the validity of arguments and configurations
    if cnn not in cnnModels:
        print(
            f"- [Warn] The given CNN model '{cnn}' is not valid (should be one of {list(cnnModels.keys())})! Setting it to 'incp3'."
        )
        cnn = "incp3"
    if packetSize <= 0 or packetSize > 50:
        print(
            f"- [Warn] The given packet size '{packetSize}' is not valid (0 < size <= 50)! Setting it to 25."
        )
        packetSize = 25
    # Fetch the list of movie frames folders (if any)
    fetchedFrames = fetchMovieFramesFolders(configs)
    if len(fetchedFrames) == 0 or not fetchedFrames:
        print(f"- [Warn] No valid frames folders found! Exiting ...")
        return
    print("- Extracting visual embeddings from the given set of movie frames ...")
    # Prepare the feature extraction model
    if cnn == "incp3":
        model = initModelInception3()
    elif cnn == "vgg19":
        model = initModelVgg19()
    else:
        print(f"- Feature extraction model '{cnn}' is not supported! Exiting ...")
        return
    if not model:
        print(
            f"- [Error] Error while initializing the feature extraction model '{cnn}'! Exiting ..."
        )
        return
    # Iterate on all frame folders in the given root directory
    for framesDir in fetchedFrames:
        # Preparing the output frames directory
        framesDir = os.path.normpath(framesDir)
        print(f"- Extracting visual embeddings from the frames in '{framesDir}' ...")
        embeddingsDir = initFeaturesPath(framesDir, featuresRootPath)
        # Skip if the output directory already exists
        if not embeddingsDir:
            continue
        # Extracting features from the frames
        modelRunner(model, framesDir, embeddingsDir, cnn, packetSize)
