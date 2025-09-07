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
    Hint: The input DataFrame should have 'user_id' and 'item_id' columns.

    Parameters
    ----------
    data: pd.DataFrame
        Given dataset in the form of a DataFrame.
    """
    # Check if required columns exist
    if "user_id" not in data.columns or "item_id" not in data.columns:
        print(
            "- [Error] The DataFrame must contain 'user_id' and 'item_id' columns. Exiting ..."
        )
        return
    # Variables
    totalInteractions = data.shape[0]
    uniqueUsers = data["user_id"].nunique()
    uniqueItems = data["item_id"].nunique()
    # Print
    print("--------------------------")
    print("- The Dataset Overview:")
    print(f"-- Total Interactions: {totalInteractions}")
    print(f"-- |U|: {uniqueUsers}")
    print(f"-- |I|: {uniqueItems}")
    print(f"-- |R|/|U|: {totalInteractions / uniqueUsers:.4f}")
    print(f"-- |R|/|I|: {totalInteractions / uniqueItems:.4f}")
    print(f"-- |R|/(|U|*|I|): {totalInteractions / (uniqueUsers * uniqueItems):.4f}")
    print("--------------------------")
