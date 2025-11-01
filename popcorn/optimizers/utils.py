import itertools
import numpy as np


def calculateKLDiv(pDist: dict, qDist: dict, eps: float = 1e-8) -> float:
    """
    Calculate the Kullback-Leibler Divergence between two probability distributions.
    [Note] It is a measure of how one probability distribution diverges from another.

    Parameters
    ----------
    pDist: dict
        The first probability distribution.
    qDist: dict
        The second probability distribution.
    eps: float, optional
        A small value to avoid division by zero, default is 1e-8.

    Returns
    -------
    float
        The Kullback-Leibler divergence between the two distributions.
        Returns 0.0 if both distributions are empty or if they are identical.
    """
    # Variables
    kldiv = 0.0
    # Check input arguments
    if not pDist and not qDist:
        print(
            "- [Warn] Both distributions are empty. Returning KL-Divergence as 0.0 ..."
        )
        return 0.0
    if not pDist or not qDist:
        print(
            "- [Warn] One of the distributions is empty. Returning KL-Divergence as inf ..."
        )
        return float("inf")
    if pDist == qDist:
        print(
            "- [Warn] Both distributions are identical. Returning KL-Divergence as 0.0 ..."
        )
        return 0.0
    try:
        # Get the unique keys from both distributions
        keys = set(pDist) | set(qDist)
        # Create probability vectors
        pVec = np.array([pDist.get(k, eps) for k in keys], dtype=float)
        qVec = np.array([qDist.get(k, eps) for k in keys], dtype=float)
        # Normalize to get valid probability distributions
        pVec /= pVec.sum()
        qVec /= qVec.sum()
        # Calculate KL-Divergence
        kldiv = (pVec * np.log(pVec / qVec)).sum()
        # Handle log(0) cases
        if np.isnan(kldiv):
            print("- [Warn] KL-Divergence is NaN. Returning as inf ...")
            return float("inf")
        # Handle division by zero cases
        if np.isinf(kldiv):
            print("- [Warn] KL-Divergence is inf. Returning as inf ...")
            return float("inf")
        # Otherwise, return the computed KL-Divergence
        return kldiv
    except Exception as e:
        print(f"- [Error] Exception occurred while calculating KL-Divergence: {e}")
        return float("inf")


def calculateDiversity(givenList: list) -> float:
    """
    Calculate the Inverse List Diversity (ILD) for a list of genres.
    [Note] It is a measure of diversity that quantifies how different the genres are from each other.

    Parameters
    ----------
    genres_list: list of list
        A list containing sublists of genres for different items.

    Returns
    -------
    float
        The ILD value, which is the average dissimilarity between pairs of genres.
        Returns 0.0 if the input list has one or fewer elements.
    """
    # Variables
    diversity = 0.0
    # Check input arguments
    if not givenList or len(givenList) <= 1:
        print(
            "- [Warn] The input list has one or fewer elements. Returning diversity as 0.0 ..."
        )
        return 0.0
    try:
        # Find all unique pairs
        pairs = itertools.combinations(givenList, 2)
        # Calculate dissimilarity for each pair
        dissimilarity = [
            1 - len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in pairs
        ]
        # Return average dissimilarity
        diversity = float(np.mean(dissimilarity))
        # Handle empty dissimilarity case
        if np.isnan(diversity):
            print("- [Warn] Diversity is NaN. Returning as 0.0 ...")
            return 0.0
        # Handle empty diversity case
        if np.isinf(diversity):
            print("- [Warn] Diversity is inf. Returning as 0.0 ...")
            return 0.0
        # Otherwise, return the computed diversity
        return diversity
    except Exception as e:
        print(f"- [Error] Exception occurred while calculating diversity: {e}")
        return 0.0


def calculateGini(values: list) -> float:
    """
    Calculate the Gini coefficient for a list of values.

    Parameters
    ----------
    values: list of float
        A list of numerical values representing a distribution.

    Returns
    -------
    gini: float
        The Gini coefficient, which ranges from 0 (perfect equality) to 1 (perfect inequality).
    """
    # Variables
    gini = 0.0
    # Check input arguments
    if not values or len(values) == 0:
        print("- [Warn] The input list is empty. Returning Gini coefficient as 0.0 ...")
        return gini
    # Sort values
    sortedValues = sorted(values)
    valueLength = len(sortedValues)
    valueSum = sum(sortedValues)
    # Handle case where all values are zero
    if valueSum == 0:
        print("- [Warn] All values are zero. Returning Gini coefficient as 0.0 ...")
        return 0.0
    # Calculate Gini coefficient
    cum = sum((i + 1) * val for i, val in enumerate(sortedValues))
    gini = (2 * cum) / (valueLength * valueSum) - (valueLength + 1) / valueLength
    return gini


def modelSelected(tag: str, model: str) -> bool:
    """
    Checks if the given tag corresponds to a model selected in the configuration.

    Parameters
    ----------
    tag: str
        The tag representing the model.
    model: str
        The model selected in the configuration.

    Returns
    -------
    bool
        True if the tag corresponds to the selected model, False otherwise.
    """
    return (
        (model == "cf" and tag in {"MF", "VAECF", "TopPop"})
        or (model == "vbpr" and tag == "VBPR")
        or (model == "vmf" and tag == "VMF")
        or (model == "amr" and tag == "AMR")
    )
