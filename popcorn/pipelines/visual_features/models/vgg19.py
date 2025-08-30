#!/usr/bin/env python3

def initModelVgg19():
    """
    Initializes the VGG-19 model for feature extraction
        - Running the example will load the VGG-19 model and download the model weights
        - The model can then be used directly to classify a photograph into one of 1,000 classes
   
    Returns
    -------
    model: Model
        The initialized Inception-v model
    """
    print("- Initializing the VGG-19 model for feature extraction ...")
    # Variables
    model = None
    try:
        # Load Inception-v3 model
        from tensorflow.keras import Model
        from tensorflow.keras.applications.vgg19 import VGG19
        # Create a model
        model = VGG19()
        # Removing the final output layer, so that the second last fully connected layer with 4,096 nodes will be the new output layer
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        print("- Initializing VGG-19 finished. Getting ready for feature extraction ...\n")
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
    from tensorflow.keras.applications.vgg19 import preprocess_input
    
    # Load VGG-19 model variables
    modelInputSize = 224 # Default input size for Inception-v3 model
    modelPreprocess = preprocess_input
    
    # Return the model variables
    return modelInputSize, modelPreprocess