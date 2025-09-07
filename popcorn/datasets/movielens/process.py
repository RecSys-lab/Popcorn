import numpy as np
import pandas as pd


def trainTestSplit(ratingsDF: pd.DataFrame, config: dict):
    """
    Splits the ratings DataFrame into training and testing sets based on the provided configuration.

    Parameters
    ----------
    ratingsDF: pd.DataFrame
        The DataFrame containing user-item interaction (ratings) data.
    config: dict
        The configuration dictionary loaded from the config.yml file.

    Returns
    -------
    trainDF: pd.DataFrame
        The training set DataFrame.
    testDF: pd.DataFrame
        The testing set DataFrame.
    """
    # Variables
    SEED = config["setup"]["seed"]
    SPLIT_MODE = config["setup"]["split"]["mode"]
    TEST_RATIO = config["setup"]["split"]["test_ratio"]
    # Split the ratings DataFrame
    print(
        f"\n- Splitting the ratings DataFrame using '{SPLIT_MODE}' mode and test ratio '{TEST_RATIO}' ..."
    )
    # Check the split mode
    if SPLIT_MODE not in ["random", "temporal", "per_user"]:
        print(f"- [Error] Unsupported split mode '{SPLIT_MODE}'! Exiting ...")
        return
    # Check the test ratio
    if TEST_RATIO <= 0 or TEST_RATIO >= 1:
        print(f"- [Warn] Test ratio should be in (0, 1)! Setting to 0.2 ...")
        TEST_RATIO = 0.2
    # Set random seed for reproducibility
    np.random.seed(SEED)
    if SPLIT_MODE == "random":
        # Randomly shuffle and split
        ratingsDF = ratingsDF.sample(frac=1, random_state=SEED).reset_index(drop=True)
        # Determine the split index
        idx = int(len(ratingsDF) * TEST_RATIO)
        # Generate train and test sets
        trainDF, testDF = ratingsDF.iloc[:-idx].copy(), ratingsDF.iloc[-idx:].copy()
    elif SPLIT_MODE == "temporal":
        # Sort by timestamp and split
        ratingsDF = ratingsDF.sort_values("timestamp")
        # Determine the split index
        idx = int(len(ratingsDF) * TEST_RATIO)
        # Generate train and test sets
        trainDF, testDF = ratingsDF.iloc[:-idx].copy(), ratingsDF.iloc[-idx:].copy()
    else:
        # Variables for 'per_user' split
        trainRec, testRec = [], []
        # For each user, put the latest interaction in the test set
        for uid, group in ratingsDF.groupby("user_id"):
            # Sort by timestamp
            group = group.sort_values("timestamp")
            # Append the last interaction to the test set, rest to train set
            testRec.append(group.iloc[-1])
            trainRec.extend(group.iloc[:-1].to_dict("records"))
        # Generate train and test sets
        trainDF, testDF = pd.DataFrame(trainRec), pd.DataFrame(testRec)
    print(
        f"- Splitting finished! Train data: {len(trainDF):,} - Test data: {len(testDF):,}."
    )
    # Return the train and test DataFrames
    return trainDF, testDF
