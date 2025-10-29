import copy
import itertools
import numpy as np


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



