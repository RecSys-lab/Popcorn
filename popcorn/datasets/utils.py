import pandas as pd


def applyKcore(dataFrame: pd.DataFrame, k: int = 0) -> pd.DataFrame:
    """
    Apply k-core filtering to the given data. This function filters the 
    DataFrame to retain only users and items that have at least k interactions.

    Parameters
    ----------
    dataFrame: pd.DataFrame
        The DataFrame containing user-item interactions.
    k: int
        The minimum number of interactions required for users and items to be retained.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only users and items with at least k interactions.
    """
    # Check if k is valid
    if k <= 0:
        print("- K-core filtering is disabled (k <= 0)! Skipping ...")
        return dataFrame
    # Variables
    changed = True
    print(f"- Applying {k}-core filtering ...")
    # Iteratively filter users and items until no more changes occur
    while changed:
        before = len(dataFrame)
        # Apply for users
        valueCount = dataFrame.user_id.value_counts()
        dataFrame = dataFrame[dataFrame.user_id.isin(valueCount[valueCount >= k].index)]
        # Apply for items
        valueCount = dataFrame.item_id.value_counts()
        dataFrame = dataFrame[dataFrame.item_id.isin(valueCount[valueCount >= k].index)]
        # Check if any changes occurred
        changed = len(dataFrame) < before
    # Return the filtered DataFrame
    return dataFrame


def printTextualDatasetStats(data: pd.DataFrame):
    """
    Print dataset statistics including total interactions, unique users, items, and interaction ratios.
    Hint: The input DataFrame should have 'userId' and 'itemId' columns.

    Parameters
    ----------
    data: pd.DataFrame
        Given dataset in the form of a DataFrame.
    """
    # Variables
    totalInteractions = data.shape[0]
    uniqueUsers = data["userId"].nunique()
    uniqueItems = data["itemId"].nunique()
    # Print
    print("--------------------------")
    print("- The Dataset Overview:")
    print(f"-- Total Interactions: {totalInteractions}")
    print(f"-- |U|: {uniqueUsers}")
    print(f"-- |I|: {uniqueItems}")
    print(f"-- |R|/|U|: {totalInteractions / uniqueUsers:.2f}")
    print(f"-- |R|/|I|: {totalInteractions / uniqueItems:.2f}")
    print(f"-- |R|/(|U|*|I|): {totalInteractions / (uniqueUsers * uniqueItems):.10f}")
    print("--------------------------")


def applyInteractionLimits(
    data: pd.DataFrame,
    minInteraction: int,
    maxInteraction: int,
    userColumnName: str = "userId",
):
    """
    Filter users based on interaction limits.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with interactions.
    minInteraction: int
        Minimum number of interactions per user.
    maxInteraction: int
        Maximum number of interactions per user.
    userColumnName: str
        Column name for users.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    # Variables
    interactionCount = data[userColumnName].value_counts()
    # Filter which users to keep
    usersToKeep = interactionCount[
        (interactionCount >= minInteraction) & (interactionCount <= maxInteraction)
    ].index
    # Return the filtered DataFrame
    return data[data[userColumnName].isin(usersToKeep)]
