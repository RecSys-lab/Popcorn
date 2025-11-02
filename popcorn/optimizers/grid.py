import math
import numpy as np
from cornac.data import Dataset


def gridMetric(
    model,
    valGrp: dict,
    trainFitSet: Dataset,
    trainSeen: dict,
    itemIdMap: dict,
    allItemIds: list,
    useGpu: bool = False,
    topN=10,
):
    """
    Computes the grid search metric for hyperparameter optimization.

    Parameters
    ----------
    model: object
        The model to be evaluated.
    valGrp: dict
        A dictionary mapping user IDs to their ground truth items.
    trainFitSet: Dataset
        The training dataset used for fitting the model.
    trainSeen: dict
        A dictionary mapping user IDs to sets of seen items during training.
    itemIdMap: dict
        A mapping of item IDs to their indices in the training dataset.
    allItemIds: list
        A list of all item IDs present in the training dataset.
    useGpu: bool
        Flag indicating whether to use GPU for hyperparameter optimization.
    topN: int, optional
        The number of top items to consider for evaluation (default is 10).

    Returns
    -------
    metric: float
        The average of recall and NDCG for the model on the validation set.
    """
    # Variables
    metric = 0.0
    rec, ndcg = [], []
    cupyAvailable = False
    # Check if GPU is available for hyperparameter optimization
    if useGpu:
        try:
            import cupy as cp

            cupyAvailable = True
            print("- [Info] CuPy enabled! Using GPU for grid search ...")
        except ImportError:
            print("- [Warning] CuPy not found! Using CPU for grid search ...")
            useGpu = False
    # Loop through each user in the validation group
    for uid, gt in valGrp.items():
        if uid not in trainFitSet.uid_map:
            continue
        # Get user index
        uidx = trainFitSet.uid_map[uid]
        # Compute scores
        scores = model.score(uidx)
        if useGpu and cupyAvailable:
            scores = cp.asarray(scores)
        # Filter seen items
        seen = trainSeen.get(uid, set())
        # Set scores of seen items to negative infinity
        cand = [(it, scores[itemIdMap[it]]) for it in allItemIds if it not in seen]
        cand.sort(key=lambda x: float(x[1]), reverse=True)
        # Get top-N recommendations
        top = [c[0] for c in cand[:topN]]
        # Compute metrics
        rec.append(len(set(top) & set(gt)) / len(gt) if gt else 0)
        dcg = sum(1 / math.log2(r + 2) for r, it in enumerate(top) if it in gt)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gt), topN)))
        ndcg.append(dcg / idcg if idcg else 0)
        # Calculate metrics
        metric = 0.5 * (np.mean(rec) + np.mean(ndcg))
        return metric


def grid():
    print(f"- Preparing GridSearch procedure ...")
