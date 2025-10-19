import os
import yaml
import json
import requests
import numpy as np
import pandas as pd


def readConfigs(configPath: str = "popcorn/config/config.yml") -> dict:
    """
    Read the configuration file and store the values in a dictionary

    Parameters
    -------
    configPath: str
        The path to the configuration file

    Returns
    -------
    windowTitle: str
        The name of the window to be shown
    """
    configs = {}
    print("- Reading the framework's configuration file ...")
    with open(configPath) as cfg:
        try:
            configs = yaml.safe_load(cfg)
            if not configs:
                print("- Error loading configuration parameters! Exiting ...")
                return
            print("- Configuration file loaded successfully!")
            return configs
        except yaml.YAMLError as err:
            print(f"[Error] Error while reading the configurations: {err}")


def parseSafe(s: str) -> np.ndarray:
    """
    Converts a string representation of a vector into a NumPy array.

    Parameters
    ----------
    s: str
        The string representation of the vector, where elements are separated by commas or spaces.

    Returns
    -------
    vec: np.ndarray
        A NumPy array containing the elements of the vector, with non-finite values replaced by 0.0.
    """
    vec = np.fromstring(str(s).replace(",", " "), sep=" ", dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def loadJsonFromUrl(jsonUrl: str) -> dict:
    """
    Load `json` data from a given URL and return it.

    Parameters
    ----------
    jsonUrl: str
        The root address to load JSON data from.

    Returns
    -------
    data: dict
        The JSON data loaded from the URL.
    """
    data = {}
    print(f"- Loading JSON data from the given URL '{jsonUrl}' ...")
    try:
        # Load JSON data from the URL
        response = requests.get(jsonUrl)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()  # Parse JSON data
        print("- JSON data loaded successfully!")
        return data
    except requests.exceptions.RequestException as e:
        print(f"- [Error] Error fetching data from {jsonUrl}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"- [Error] Error parsing JSON data: {e}")
        return None


def loadJsonFromFilePath(jsonPath: str):
    """
    Load `json` data from a given file path and return it.

    Parameters:
        jsonPath (str): The path to the JSON file.

    Returns:
        dict: The JSON data loaded from the file.
    """
    try:
        # Check if the file exists
        if not os.path.exists(jsonPath):
            raise FileNotFoundError(
                f"- [Error] File '{jsonPath}' not found! Exiting ..."
            )
        # Load the JSON data
        with open(jsonPath, "r") as jsonFile:
            jsonData = json.load(jsonFile)
        return jsonData
    except Exception as e:
        print(f"- [Error] An error occurred while loading the JSON data: {e}")
        return None


def serializeListColumn(dataFrame: pd.DataFrame, columnName: str) -> pd.Series:
    """
    Serialize a list column in a pandas DataFrame into a string representation.
    This avoids truncation issues when saving to CSV by converting lists to comma-separated strings.

    Parameters
    ----------
    dataFrame: pd.DataFrame
        The input DataFrame containing the list column
    columnName: str
        The name of the column to serialize

    Returns
    -------
    pd.Series
        A pandas Series containing the serialized list column
    """
    return dataFrame[columnName].apply(
        lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
    )
