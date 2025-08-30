#!/usr/bin/env python3

def initModelInception3():
    """
    Initializes the Inception-v3 (GoogleNet) model for feature extraction
        - The model expects color images to have the square shape 299 x 299
        - Running the example will load the Inception-v3 model and download the model weights
    
    Returns
    -------
    model: Model
        The initialized Inception-v model
    """
    print("- Initializing the Inception-v3 model for feature extraction ...")
    # Variables
    model = None
    try:
        # Load Inception-v3 model
        from tensorflow.keras import Model
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        # Create a model
        model = InceptionV3()
        # Removing the final output layer, so that the second last fully connected layer with 2,048 nodes will be the new output layer
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        print("- Initializing Inception-v3 finished. Getting ready for feature extraction ...\n")
        # Return the model
        return model
    except Exception as otherError:
        print(f'Error while processing video frames: {str(otherError)}')
        return None

def getModelVariables():
    """
    Provides the model variables for the VGG-19 model
   
    Returns
    -------
    modelInputSize: int
        The input size for the VGG-19 model
    modelPreprocess: function
        The preprocess function for the VGG-19 model
    """
    # Load proper imports
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    # Load Inception-v3 model variables
    modelInputSize = 299  # Default input size for Inception-v3 model
    modelPreprocess = preprocess_input
    
    # Return the model variables
    return modelInputSize, modelPreprocess