from popcorn.optimizers.grid import grid
from popcorn.optimizers.utils import modelSelected
from cornac.models import MF, VBPR, VMF, AMR, VAECF


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
