def getParametersGrid(isFastPrtye: bool = True, nEpochs: int = 10):
    """
    Get the parameter grid for different optimizers.

    Parameters
    ----------
    isFastPrtye: bool, optional
        Flag to indicate if a fast prototype is being used. Default is True.
    nEpochs: int, optional
        The number of epochs to use for training. Default is 10.

    Returns
    -------
    parametersGrid: dict
        A dictionary containing the parameter grid for each optimizer.
    """
    print(
        f"- Preparing parameter grid for optimizers (prototype='{isFastPrtye}', epochs={nEpochs}) ..."
    )
    # Define parameter grids
    GR_MF = [
        {"k": k, "learning_rate": lr, "lambda_reg": 0.01, "max_iter": 50}
        for k in (32, 64, 128)
        for lr in (0.01, 0.005)
    ][0:5]
    GR_VAECF = [
        {"k": k, "learning_rate": lr, "beta": 0.01}
        for k in (32, 64, 128)
        for lr in (0.001, 0.0005)
    ][0:5]
    if isFastPrtye:
        GR_VBPR = [
            {
                "k": k,
                "k2": k2,
                "learning_rate": lr,
                "lambda_w": 0.01,
                "lambda_b": 0.01,
                "n_epochs": 1,
            }
            for k in (32, 64, 128)
            for k2 in (8, 16)
            for lr in (0.001,)
        ][0:5]
    else:
        GR_VBPR = [
            {
                "k": k,
                "k2": k2,
                "learning_rate": lr,
                "lambda_w": 0.01,
                "lambda_b": 0.01,
                "n_epochs": nEpochs,
            }
            for k in (32, 64, 128)
            for k2 in (8, 16)
            for lr in (0.001,)
        ][0:5]
    if isFastPrtye:
        GR_VMF = [
            {"k": k, "learning_rate": lr, "n_epochs": 1}
            for k in (32, 64, 128)
            for lr in (0.01,)
        ][0:5]
    else:
        GR_VMF = [
            {"k": k, "learning_rate": lr, "n_epochs": nEpochs}
            for k in (32, 64, 128)
            for lr in (0.01,)
        ][0:5]
    if isFastPrtye:
        GR_AMR = [
            {"k": k, "k2": k2, "learning_rate": lr, "n_epochs": 1}
            for k in (32, 64, 128)
            for k2 in (16, 32)
            for lr in (0.001,)
        ][0:5]
    else:
        GR_AMR = [
            {"k": k, "k2": k2, "learning_rate": lr}
            for k in (32, 64, 128)
            for k2 in (16, 32)
            for lr in (0.001,)
        ][0:5]
    # Assemble parameter grid
    parametersGrid = {
        "MF": GR_MF,
        "VAECF": GR_VAECF,
        "VBPR": GR_VBPR,
        "VMF": GR_VMF,
        "AMR": GR_AMR,
    }
    return parametersGrid
