import math
import time
import inspect
import numpy as np
from cornac.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from popcorn.optimizers.utils import fitModalities


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


def grid(
    dataDict: dict, cornacModel, name: str, scenario: str, paramGrid: dict, *fit_args
) -> dict:
    """
    Performs hyperparameter optimization (HPO) using grid search for a given model class.

    Parameters
    ----------
    dataDict: dict
        A dictionary containing the data to be used for hyperparameter optimization.
    cornacModel: class
        The Cornac model class for which to perform hyperparameter optimization.
    name: str
        The name of the model being optimized.
    scenario: str
        The scenario or modality for which the model is being optimized.
    paramGrid: dict
        A dictionary containing the hyperparameter grid to search over.
    fit_args: tuple
        Additional arguments to be passed to the model fitting function.

    Returns
    -------
    """
    # Variables
    start = time.time()
    config = dataDict["config"]
    valGrp = dataDict["val_grp"]
    itemIdMap = dataDict["iid_map"]
    allItemIds = dataDict["all_iids"]
    trainSeen = dataDict["train_seen"]
    trainFitSet = dataDict["train_fit_set"]
    # Config Variables
    seed = config["setup"]["seed"]
    useGPU = config["setup"]["use_gpu"]
    parallelHPO = config["setup"]["use_parallel"]
    print(
        f"- Starting GridSearch procedure (seed: {seed}, useGPU: {useGPU}, parallelHPO: {parallelHPO})..."
    )

    # Evaluation function
    def gridEvalCalculator(params: dict):
        # Make a copy of parameters
        paramCopy = params.copy()
        # Fit with modalities if needed
        if useGPU and "use_gpu" in inspect.signature(cornacModel).parameters:
            paramCopy["use_gpu"] = True
        # Fit model
        model = cornacModel(seed=seed, **paramCopy)
        # fitModalities expects (model, baseDataset, imgModality=None, featModality=None)
        fitModalities(model, trainFitSet, *fit_args)
        # Compute metric
        smetric = gridMetric(
            model, valGrp, trainFitSet, trainSeen, itemIdMap, allItemIds, useGPU
        )
        print(f"-- Fitting '{paramCopy}' to get {smetric:.4f} ...")
        return smetric, model, paramCopy

    # Perform grid search
    print(
        f"-- HPO '{name}' in scenario '{scenario}' with {len(paramGrid)} param sets..."
    )
    if parallelHPO and len(paramGrid) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(paramGrid))) as ex:
            results = list(ex.map(gridEvalCalculator, paramGrid))
    else:
        results = [gridEvalCalculator(p) for p in paramGrid]
    # Get best result
    best = max(results, key=lambda x: x[0])
    print(
        f"-- Found the best case: '{name}' in '{scenario}', item '{best[2]}' ({best[0]:.4f}), done in {time.time()-start:.1f}s."
    )
    return best[1], best[2]
