import copy
import itertools
import numpy as np

# Multimodal fusion variants
MULTI_VARIANTS = [
    ("concat", None),
    ("pca", 0.95),
    ("cca", 40),
]

def parseEmbedding(givenEmbedding):
    """
    Parses the embedding string or list into a numpy array. It can work with textual of visual embeddings.

    Parameters
    ----------
    givenEmbedding: str or list
        Given embedding in the form of a string or list.
    
    Returns
    -------
    np.ndarray
        Parsed embedding as a numpy array.
    """
    # Check the type of the given embedding
    if isinstance(givenEmbedding, str):
        arr = [float(x) for x in givenEmbedding.strip().split()]
        return np.array(arr, dtype=np.float32)
    elif isinstance(givenEmbedding, (list, np.ndarray)):
        return np.array(givenEmbedding, dtype=np.float32)
    else:
        return None

def modelIsSelected(tag: str, model: str) -> bool:
    """
    Checks if the given tag corresponds to a model selected in the configuration.

    Parameters
    ----------
    tag : str
        The tag representing the model.
    """
    return (
        (model == "cf" and tag in {"MF", "VAECF", "TopPop"})
        or (model == "vbpr" and tag == "VBPR")
        or (model == "vmf" and tag == "VMF")
        or (model == "amr" and tag == "AMR")
    )

def fitWithModalities(model, base_ds, item_img=None, item_feat=None):
    """
    Fits the model with the given dataset and item modalities.

    Parameters
    ----------
    model : object
        The model to be fitted.
    base_ds : Dataset
        The base dataset containing user-item interactions.
    item_img : np.ndarray, optional
        The item image features to be used in the dataset.
    item_feat : np.ndarray, optional
        The item feature embeddings to be used in the dataset.
    """
    ds = copy.deepcopy(base_ds)
    if item_img is not None:
        ds.item_image = item_img
    if item_feat is not None:
        ds.item_feature = item_feat
    model.fit(ds)


def gini(x):
    """
    Calculate the Gini coefficient for a list of values.
    The Gini coefficient is a measure of statistical dispersion that represents the income or wealth distribution of a nation's residents, and is often used as a measure of inequality.

    Parameters
    ----------
    x : list or np.ndarray
        A list or array of numerical values.

    Returns
    -------
    float
        The Gini coefficient, which ranges from 0 (perfect equality) to 1 (perfect inequality).
    """
    if not x:
        return 0.0
    sx = sorted(x)
    n = len(sx)
    tot = sum(sx)
    if tot == 0:
        return 0.0
    cum = sum((i + 1) * val for i, val in enumerate(sx))
    return (2 * cum) / (n * tot) - (n + 1) / n


def ild(genres_list):
    """
    Calculate the Inverse List Diversity (ILD) for a list of genres.
    The ILD is a measure of diversity that quantifies how different the genres are from each other.

    Parameters
    ----------
    genres_list : list of list
        A list containing sublists of genres for different items.

    Returns
    -------
    float
        The ILD value, which is the average dissimilarity between pairs of genres.
        Returns 0.0 if the input list has one or fewer elements.
    """
    if len(genres_list) <= 1:
        return 0.0
    pairs = itertools.combinations(genres_list, 2)
    dissim = [1 - len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in pairs]
    return float(np.mean(dissim))


def kl_div(p, q, eps=1e-8):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.
    The Kullback-Leibler divergence is a measure of how one probability distribution diverges from a second, expected probability distribution.

    Parameters
    ----------
    p : dict
        The first probability distribution, represented as a dictionary where keys are events and values are their probabilities.
    q : dict
        The second probability distribution, represented as a dictionary where keys are events and values are their probabilities.
    eps : float, optional
        A small value to avoid division by zero, default is 1e-8.

    Returns
    -------
    float
        The Kullback-Leibler divergence between the two distributions.
        Returns 0.0 if both distributions are empty or if they are identical.
    """
    keys = set(p) | set(q)
    p_vec = np.array([p.get(k, eps) for k in keys], dtype=float)
    q_vec = np.array([q.get(k, eps) for k in keys], dtype=float)
    p_vec /= p_vec.sum()
    q_vec /= q_vec.sum()
    return (p_vec * np.log(p_vec / q_vec)).sum()