import pandas as pd

def filterRatingsByUserInteraction(
    ratingsDF: pd.DataFrame,
    minInteraction: int,
    maxInteraction: int,
    userColumn: str = "user_id",
):
    """
    Filter items with a particular range of user interactions.

    Parameters
    ----------
    ratingsDF: pd.DataFrame
        DataFrame with ratings (interactions).
    minInteraction: int
        Minimum number of interactions per user.
    maxInteraction: int
        Maximum number of interactions per user.
    userColumn: str
        Column name for users.

    Returns
    -------
    ratingsDF_filtered: pd.DataFrame
        Filtered ratings DataFrame.
    """
    # Check if ratingsDF is empty
    if ratingsDF is None or ratingsDF.empty:
        print("- [Error] The input DataFrame is empty! Exiting ...")
        return None
    # Check if user column exists
    if userColumn not in ratingsDF.columns:
        print(
            f"- [Error] The specified user column '{userColumn}' does not exist in the DataFrame! Exiting ..."
        )
        return None
    # Check interaction limits
    if minInteraction < 0 or maxInteraction < 0 or minInteraction > maxInteraction:
        print(
            "- [Error] Invalid interaction limits! Ensure '0 <= minInteraction <= maxInteraction'. Exiting ..."
        )
        return None
    # Variables
    totalInteractions = ratingsDF[userColumn].value_counts()
    print(
        f"- Applying interaction limits: min={minInteraction}, max={maxInteraction} ..."
    )
    # Filter which users to keep
    usersToKeep = totalInteractions[
        (totalInteractions >= minInteraction) & (totalInteractions <= maxInteraction)
    ].index
    # Filtered DataFrame
    ratingsDF_filtered = ratingsDF[ratingsDF[userColumn].isin(usersToKeep)]
    print(f"- Kept {len(ratingsDF_filtered)} rows with {minInteraction} to {maxInteraction} interactions.")
    # Return the filtered DataFrame
    return ratingsDF_filtered