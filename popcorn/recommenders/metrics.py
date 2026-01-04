import pandas as pd


def calculateMeanRecMetrics(recs: pd.DataFrame, top_n: int) -> list[dict]:
    """
    Calculate average metrics from the recommendation records.
    This function computes various metrics such as Recall, NDCG, Cold-start Rate,
    Coverage, Popularity Bias, Fairness, Novelty, Diversity, and Calibration Bias
    for each model and scenario in the recommendation records.

    Parameters
    ----------
    recs: pd.DataFrame
        A DataFrame containing recommendation records with columns for various metrics.
    top_n: int
        The number of top recommendations considered for metric calculations.

    Returns
    -------
    list of dict
        A list of dictionaries where each dictionary contains the average metrics
        for a specific model and scenario.
    """
    # Variables
    metricRows = []
    # Check the arguments
    if recs.empty:
        print(
            "- [Warn] Empty recommendation records provided. Returning empty metrics list ..."
        )
        return metricRows
    # Calculate average metrics for each model and scenario
    for col in [c for c in recs.columns if c.startswith("rec_")]:
        model, scenario = col.split("_", 2)[1:]
        ndcg = recs[f"ND_{model}_{scenario}"].mean()
        recall = recs[f"RC_{model}_{scenario}"].mean()
        popBias = recs[f"PB_{model}_{scenario}"].mean()
        novelty = recs[f"NO_{model}_{scenario}"].mean()
        calBias = recs[f"CB_{model}_{scenario}"].mean()
        fairness = recs[f"FA_{model}_{scenario}"].mean()
        coldrate = recs[f"CR_{model}_{scenario}"].mean()
        coverage = recs[f"CV_{model}_{scenario}"].mean()
        diversity = recs[f"DI_{model}_{scenario}"].mean()
        # Append the metrics for the current model and scenario
        metricRows.append(
            {
                "model": model,
                f"NDCG@{top_n}": ndcg,
                "Novelty": novelty,
                f"Recall@{top_n}": recall,
                "scenario": scenario,
                "Fairness": fairness,
                "Diversity": diversity,
                f"Coverage@{top_n}": coverage,
                f"ColdRate@{top_n}": coldrate,
                "PopularityBias": popBias,
                "CalibrationBias": calBias,
            }
        )
    return metricRows
