import os
import math
import cornac
import itertools
import collections
import numpy as np
import pandas as pd
from popcorn.recommenders.utils import SUPPORTED_TOP_N
from popcorn.recommenders.metrics import calculateMeanRecMetrics
from popcorn.datasets.poison_rag_plus.utils import SUPPORTED_LLMS
from popcorn.optimizers.utils import (
    calculateDcg,
    calculateGini,
    calculateIdcg,
    calculateNdcg,
    calculateKLDiv,
    calculateDiversity,
)


def getTopN(
    model,
    uid,
    N: int,
    trainSet: cornac.data.Dataset,
    itemIdMap: dict,
    allItemIds: list,
    trainSeen: dict,
):
    """
    Get the top N recommendations for a user based on the model's scores.

    Parameters
    ----------
    model: object
        The recommendation model used to score items.
    uid: int
        The user ID for whom recommendations are to be generated.
    N: int
        The number of top recommendations to return.
    trainSet: object
        The training set containing user-item interactions.
    itemIdMap: dict
        A mapping from item IDs to their indices in the model.
    allItemIds: list
        A list of all item IDs available for recommendation.
    trainSeen: dict
        A dictionary mapping user IDs to sets of items they have already interacted with.

    Returns
    -------
    list
        A list of the top N item IDs recommended for the user.
    """
    # If the user is not in the training set, return an empty list
    if uid not in trainSet.uid_map:
        return []
    # Score all items for the user
    scores = model.score(trainSet.uid_map[uid])
    # Filter out items the user has already seen
    cand = [
        (it, scores[itemIdMap[it]])
        for it in allItemIds
        if it not in trainSeen.get(uid, set())
    ]
    # Sort candidates by score in descending order and select top N
    cand.sort(key=lambda x: float(x[1]), reverse=True)
    # Return the top N item IDs
    return [c[0] for c in cand[:N]]


def generateLists(
    config: dict,
    trainDF: pd.DataFrame,
    trainSet: cornac.data.Dataset,
    testDF: pd.DataFrame,
    genreDict: dict,
    finalModels: dict,
):
    """
    Generates recommendation lists for users in the test set using the provided models,
    and calculates various recommendation metrics.

    Parameters
    ----------
    config: dict
        The configuration dictionary containing experiment settings.
    trainDF: pd.DataFrame
        The training DataFrame containing user-item interactions.
    trainSet: Dataset
        The training dataset object containing user-item interactions.
    testDF: pd.DataFrame
        The testing DataFrame containing user-item interactions.
    genreDict: dict
        A dictionary mapping item IDs to their genres.
    finalModels: dict
        A dictionary containing the final recommendation models.

    Returns
    -------
    """
    print("- Generating recommendation lists ...")
    # Variables
    rows = []
    trainPop = {}
    TOP_N = config["recommender"]["top_n"]
    ROOT_PATH = config["general"]["root_path"]
    MODEL_CHOICE = config["setup"]["model_choice"]
    OUTPUT_PATH = config["general"]["output_path"]
    COLD_START_THRESHOLD = config["recommender"]["cold_threshold"]
    LLM = config["datasets"]["unimodal"]["poison_rag_plus"]["llm"]
    ML_VERSION = config["datasets"]["unimodal"]["movielens"]["version"]
    AUGMENTED = (
        "aug"
        if config["datasets"]["unimodal"]["poison_rag_plus"]["augmented"]
        else "raw"
    )
    # Some checks
    if LLM not in SUPPORTED_LLMS:
        print(f"- [Error] Unsupported LLM backbone '{LLM}'! Exiting ...")
        return
    if TOP_N not in SUPPORTED_TOP_N:
        print(f"- [Warn] Unsupported top-N value '{TOP_N}'! Setting to 10 ...")
        TOP_N = 10
    if (
        COLD_START_THRESHOLD is None
        or not isinstance(COLD_START_THRESHOLD, int)
        or COLD_START_THRESHOLD < 0
    ):
        print(
            f"- [Warn] Invalid cold-start threshold '{COLD_START_THRESHOLD}'! Setting to 5 ..."
        )
        COLD_START_THRESHOLD = 5
    # Prepare item ID mappings
    trainSeen = trainDF.groupby("user_id")["item_id"].apply(set).to_dict()
    allItemIds, itemIdMap = trainSet.item_ids, trainSet.iid_map
    # Prepare popularity of items
    for _, itemId in trainSet.user_data.items():
        for ii in itemId[0]:
            trainPop[ii] = trainPop.get(ii, 0) + 1
    # Get max popularity
    maxPopularity = max(trainPop.values())
    # Prepare cold items set
    coldItems = {i for i, c in trainPop.items() if c <= COLD_START_THRESHOLD}
    # Prepare coverage dictionary
    coverageDict = collections.defaultdict(set)
    # Generate per-user recommendations
    for uid, grp in testDF.groupby("user_id"):
        # Ground truth and training items
        gt = set(grp.item_id.tolist())
        # Train items
        trainItems = trainSeen.get(uid, set())
        # User genres
        userGenres = list(
            itertools.chain(*(genreDict.get(str(it), []) for it in trainItems))
        )
        # User genre distribution
        userGenreDist = pd.Series(userGenres).value_counts().to_dict()
        # Prepare the row
        r = {"user_id": uid, "train": list(trainItems), "gt": list(gt)}
        # Prepare recommendations for each model and scenario
        for (mdl, scn), mod in finalModels.items():
            # Get top-N recommendations
            rec = getTopN(mod, uid, TOP_N, trainSet, itemIdMap, allItemIds, trainSeen)
            # Recommendation list
            r[f"rec_{mdl}_{scn}"] = rec
            # Cold-start Rate
            coldRate = sum(it in coldItems for it in rec) / len(rec) if rec else 0
            r[f"CR_{mdl}_{scn}"] = coldRate
            coverageDict[(mdl, scn)].update(rec)
            # Popularity Bias, Fairness, Novelty, Diversity, Calibration Bias
            popularityBias = (
                np.mean([trainPop.get(it, 0) / maxPopularity for it in rec])
                if rec
                else 0
            )
            fairness = 1 - calculateGini([trainPop.get(it, 0) for it in rec])
            novelty = (
                np.mean(
                    [
                        -math.log2(trainPop.get(it, 1) / len(trainSet.user_data))
                        for it in rec
                    ]
                )
                if rec
                else 0
            )
            # Diversity, Calibration Bias, NDCG, Recall
            recGenre = [genreDict.get(str(it), ["(none)"]) for it in rec]
            diversity = calculateDiversity(recGenre)
            recGenreFlat = list(itertools.chain(*recGenre))
            recGenreDict = pd.Series(recGenreFlat).value_counts().to_dict()
            calib = calculateKLDiv(userGenreDist, recGenreDict)
            dcg = calculateDcg(rec, gt)
            idcg = calculateIdcg(gt)
            ndcg = calculateNdcg(dcg, idcg)
            recall = len(set(rec) & gt) / len(gt) if gt else 0
            # Update the row with the calculated metrics
            r.update(
                {
                    f"ND_{mdl}_{scn}": ndcg,
                    f"CB_{mdl}_{scn}": calib,
                    f"RC_{mdl}_{scn}": recall,
                    f"NO_{mdl}_{scn}": novelty,
                    f"FA_{mdl}_{scn}": fairness,
                    f"DI_{mdl}_{scn}": diversity,
                    f"PB_{mdl}_{scn}": popularityBias,
                }
            )
        # Append the row
        rows.append(r)
    # Convert to DataFrame and save results
    recs = pd.DataFrame(rows)
    # Broadcast coverage columns
    for (mdl, scn), items in coverageDict.items():
        coverage = len(items) / len(allItemIds)
        recs[f"CV_{mdl}_{scn}"] = coverage
    # Save recommendation lists and metrics
    suffix = f"ml{ML_VERSION}_{MODEL_CHOICE}_{LLM}_{AUGMENTED}"
    outputSavePath = (
        os.path.join(ROOT_PATH, "outputs") if OUTPUT_PATH == "" else OUTPUT_PATH
    )
    if not os.path.exists(outputSavePath):
        os.mkdir(outputSavePath)
    reclistSavePath = os.path.join(outputSavePath, f"reclist_{suffix}.csv")
    recs.to_csv(reclistSavePath, index=False)
    print(
        f"- Recommendation lists saved have been saved in '{reclistSavePath}'! Samples:"
    )
    print(recs.head(3))
    # Calculate and save metrics
    metricsSavePath = os.path.join(outputSavePath, f"metrics_{suffix}.csv")
    metricRows = calculateMeanRecMetrics(recs, TOP_N)
    aggMetrics = pd.DataFrame(metricRows)
    aggMetrics.to_csv(metricsSavePath, index=False)
    print(
        f"- Recommendation lists saved have been saved in '{metricsSavePath}'! Samples:"
    )
    pd.options.display.float_format = lambda x: f"{x:8.3f}"
    print(aggMetrics.sort_values(["model", "scenario"]).to_string(index=False))
