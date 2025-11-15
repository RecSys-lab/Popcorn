import scipy.sparse
import pandas as pd
from cornac.data import Dataset
from sklearn.model_selection import train_test_split
from popcorn.optimizers.parameters import getParametersGrid
from popcorn.optimizers.hpo import applyHyperparameterOptimization, refitBestModels


def gridSearch(
    config: dict, trainDF: pd.DataFrame, trainSet: pd.DataFrame, modalitiesDict: dict
):
    """
    Performs hyperparameter optimization (HPO) using grid search for multiple models.
    This function prepares the dataset, defines the hyperparameter grid for each model,
    and runs the grid search to find the best configurations.

    Parameters
    ----------
    config: dict
        The configuration dictionary containing experiment settings.
    trainDF: pd.DataFrame
        The training DataFrame containing user-item interactions.
    trainSet: Dataset
        The training dataset object containing user-item interactions.
    modalitiesDict: dict
        A dictionary containing modalities for different models.

    Returns
    -------
    finalModels: dict
        A dictionary containing the final models after hyperparameter tuning.
    """
    # Variables
    finalModels = {}
    seed = config["setup"]["seed"]
    N_EPOCHS = config["setup"]["n_epochs"]
    MODEL_CHOICE = config["setup"]["model_choice"]
    testRatio = config["setup"]["split"]["test_ratio"]
    isFastPrtye = config["setup"]["is_fast_prototype"]
    # Check arguments
    if trainDF is None:
        print("- [Error] Training DataFrame is missing. Exiting grid search ...")
        return finalModels
    if trainSet is None:
        print("- [Error] Training Dataset is missing. Exiting grid search ...")
        return finalModels
    if modalitiesDict is None:
        print("- [Error] Modalities dictionary is missing. Exiting grid search ...")
        return finalModels
    # Monkey‑patch to add .A property to scipy sparse matrices
    if not hasattr(scipy.sparse.csr_matrix, "A"):
        scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
    # Prepare dataset
    print(f"- Preparing GridSearch procedure ...")
    trainFitDF, valDF = train_test_split(
        trainDF, test_size=testRatio, random_state=seed
    )
    valGrp = valDF.groupby("user_id")["item_id"].apply(list).to_dict()
    trainSeen = trainFitDF.groupby("user_id")["item_id"].apply(set).to_dict()
    trainFitSet = Dataset.from_uir(
        trainFitDF[["user_id", "item_id", "rating"]].values.tolist()
    )
    allItemIds, itemIdMap = trainFitSet.item_ids, trainFitSet.iid_map
    print(f"- Training and validation sets prepared!")
    # Keep them in a dictionary
    dataDict = {
        "val_grp": valGrp,
        "train_seen": trainSeen,
        "iid_map": itemIdMap,
        "all_iids": allItemIds,
        "train_fit_set": trainFitSet,
        "config": config,
    }
    # Get parameter grid
    if N_EPOCHS is None or not isinstance(N_EPOCHS, int) or N_EPOCHS <= 0 or N_EPOCHS > 100:
        print(
            f"- [Warning] Invalid number of epochs {N_EPOCHS}. Using default of 10 epochs."
        )
        N_EPOCHS = 10
    parametersGrid = getParametersGrid(isFastPrtye, N_EPOCHS)
    # Apply HPO for the models
    modelsCfg = applyHyperparameterOptimization(
        MODEL_CHOICE, parametersGrid, dataDict, modalitiesDict
    )
    # Re-fit the best models on the full training set
    finalModels = refitBestModels(
        trainSet, modalitiesDict, modelsCfg, MODEL_CHOICE, config
    )
    print(f"- Grid search done! Kept {len(finalModels)} final models.")
    return finalModels
