import pickle
import pandas as pd
from pathlib import Path

def load_pickle(path: str):
    """
    Load and return data from a pickle file.

    Parameters
    ----------
    path : str
        Path to the .pkl file

    Returns
    -------
    Any
        Python object stored in the pickle file
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data

def dict_to_df(data: dict) -> pd.DataFrame:
    """
    Convert a dictionary into a pandas DataFrame.

    Supports:
        - dict of lists  -> columns
        - dict of dicts  -> rows
        - list of dicts  -> rows

    Parameters
    ----------
    data : dict
        Dictionary loaded from pickle

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data

    # Dict of lists or list of dicts
    try:
        return pd.DataFrame(data)
    except Exception:
        pass

    # Dict of dicts
    try:
        return pd.DataFrame.from_dict(data, orient="index")
    except Exception:
        pass

    raise ValueError("Unsupported dictionary format for DataFrame conversion.")

def load_all_pickles(directory: str | Path) -> list:
    """
    Load all pickle files from a directory into a list.

    Parameters
    ----------
    directory : str | Path
        Path to folder containing .pkl files

    Returns
    -------
    list
        List of objects loaded from pickle files
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    objects = []

    for pkl_path in sorted(directory.glob("*.pkl")):
        obj = load_pickle(pkl_path)
        objects.append(obj)

    return objects