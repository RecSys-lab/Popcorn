import inspect
import pandas as pd
from popcorn.optimizers.grid import grid
from cornac.models import MF, VBPR, VMF, AMR, VAECF, MostPop
from popcorn.optimizers.utils import modelSelected, fitModalities


def applyHyperparameterOptimization(
    model: str, parametersGrid: dict, dataDict: dict, modalitiesDict: dict
):
    """
    Apply hyperparameter optimization to the given model using the specified parameter grid.

    Parameters
    ----------
    model: str
        The model choice for which to apply hyperparameter optimization.
    parametersGrid: dict
        A dictionary containing the hyperparameter grids for different models.
    dataDict: dict
        A dictionary containing the prepared data for training and validation.
    modalitiesDict: dict
        A dictionary containing modalities for different models.

    Returns
    -------
    modelsCfg: dict
        A dictionary containing the best model configurations after hyperparameter optimization.
    """
    # Variables
    modelsCfg = {}
    print("- Starting HPO ...")
    # MF
    if modelSelected("MF", model):
        modelsCfg["MF"] = grid(dataDict, MF, "MF", "(na)", parametersGrid["MF"])
    # VAECF
    if modelSelected("VAECF", model):
        modelsCfg["VAECF"] = grid(
            dataDict, VAECF, "VAECF", "(na)", parametersGrid["VAECF"]
        )
    # VBPR
    if modelSelected("VBPR", model):
        for mod in ("visual", "audio", "text"):
            modelsCfg[f"VBPR_{mod}"] = grid(
                dataDict,
                VBPR,
                "VBPR",
                mod,
                parametersGrid["VBPR"],
                modalitiesDict["concat"][f"{mod}_image"],
            )
        for mv in modalitiesDict:
            if mv == "concat":
                continue
            modelsCfg[f"VBPR_{mv}"] = grid(
                dataDict,
                VBPR,
                "VBPR",
                mv,
                parametersGrid["VBPR"],
                modalitiesDict[mv]["all_image"],
            )
    # VMF
    if modelSelected("VMF", model):
        for mod in ("visual", "audio", "text"):
            modelsCfg[f"VMF_{mod}"] = grid(
                dataDict,
                VMF,
                "VMF",
                mod,
                parametersGrid["VMF"],
                modalitiesDict["concat"][f"{mod}_image"],
            )
        for mv in modalitiesDict:
            if mv == "concat":
                continue
            modelsCfg[f"VMF_{mv}"] = grid(
                dataDict,
                VMF,
                "VMF",
                mv,
                parametersGrid["VMF"],
                modalitiesDict[mv]["all_image"],
            )
    # AMR
    if modelSelected("AMR", model):
        for mod in ("visual", "audio", "text"):
            modelsCfg[f"AMR_{mod}"] = grid(
                dataDict,
                AMR,
                "AMR",
                mod,
                parametersGrid["AMR"],
                modalitiesDict["concat"][f"{mod}_image"],
                modalitiesDict["concat"]["all_feature"],
            )
        for mv in modalitiesDict:
            if mv == "concat":
                continue
            modelsCfg[f"AMR_{mv}"] = grid(
                dataDict,
                AMR,
                "AMR",
                mv,
                parametersGrid["AMR"],
                modalitiesDict[mv]["all_image"],
                modalitiesDict[mv]["all_feature"],
            )
    print(f"- HPO done! Kept {len(modelsCfg)} configs.")
    return modelsCfg


def refitBestModels(
    trainSet: pd.DataFrame,
    modalitiesDict: dict,
    modelsCfg: dict,
    modelChoice: str,
    cfg: dict,
) -> dict:
    """
    Re-fits the best models on the full training set after hyperparameter optimization.

    Parameters
    ----------
    trainSet: pd.DataFrame
        The full training dataset.
    modalitiesDict: dict
        A dictionary containing modalities for different models.
    modelsCfg: dict
        A dictionary containing the best model configurations after hyperparameter optimization.
    modelChoice: str
        The model choice for which to re-fit the best models.
    cfg: dict
        The configuration dictionary containing experiment settings.
    """
    # Variables
    finalModels = {}
    seed = cfg["setup"]["seed"]
    useGpu = cfg["setup"]["use_gpu"]
    # Re-fit models
    if modelSelected("TopPop", modelChoice):
        mpop = MostPop()
        mpop.fit(trainSet)
        finalModels[("TopPop", "NA")] = mpop
    for tag, (bestModel, cfg) in modelsCfg.items():
        model, variant = tag.split("_", 1) if "_" in tag else (tag, "NA")
        extras = {}
        if useGpu and "use_gpu" in inspect.signature(bestModel.__class__).parameters:
            extras["use_gpu"] = True
        if model in {"MF", "VAECF"}:
            new = bestModel.__class__(seed=seed, **cfg, **extras)
            fitModalities(new, trainSet)
            finalModels[(model, variant)] = new
        else:
            if variant in ("visual", "audio", "text"):
                img = modalitiesDict["concat"][f"{variant}_image"]
                feat = None
            else:
                img = modalitiesDict[variant]["all_image"]
                feat = modalitiesDict[variant].get("all_feature")
            new = bestModel.__class__(seed=seed, **cfg, **extras)
            fitModalities(new, trainSet, img, feat)
            finalModels[(model, variant)] = new
    print(f"- Re-fit finished for '{model}' with variant '{variant}'.")
    return finalModels
