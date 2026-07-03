from dataclasses import fields
import pickle
from attrs import fields
import pandas as pd
from pathlib import Path
import numpy as np

def load_met4a_pickle(path: str):
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

def load_met4a_npz(path: str | Path) -> dict:
    """
    Load a MET4A .npz file written by station_runner.
    Returns
    -------
    dict
        Dictionary containing metadata plus a recarray named 'samples'.
    """

    archive = np.load(path)

    return {
        "start_time": float(archive["start_time"]),
        "ref_pressure": float(archive["ref_pressure"]),
        "scale_fac": float(archive["scale_fac"]),
        "nsamp": int(archive["nsamp"]),
        "samples": archive["samples"].view(np.recarray),
    }

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
        return data  # already a dataframe

    # Case 1: dict of lists (most common)
    try:
        return pd.DataFrame(data)
    except Exception:
        pass

    # Case 2: dict of dicts
    try:
        return pd.DataFrame.from_dict(data, orient="index")
    except Exception:
        pass

    raise ValueError("Unsupported dictionary format for DataFrame conversion.")

def load_all_met4a_pickles(directory: str | Path) -> list:
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
        obj = load_met4a_pickle(pkl_path)
        objects.append(obj)

    return objects

def load_all_met4a_npz(directory: str | Path) -> list[dict]:

    """Load all MET4A .npz files from a directory."""

    directory = Path(directory)

    if not directory.exists():

        raise FileNotFoundError(f"Directory not found: {directory}")

    return [

        load_met4a_npz(path)

        for path in sorted(directory.glob("*.met4a.npz"))

    ]

def convert_to_datetime(start_time, offsets):
    """
    Convert start time and offsets to a pandas DatetimeIndex.

    Parameters
    ----------
    start_time : int or float
        Start time in seconds since epoch
    offsets : list or array-like
        List of time offsets in seconds to add to the start time

    Returns
    -------
    time_axis : pd.DatetimeIndex
        DatetimeIndex representing the time axis for the data
    """
    start_dt = pd.to_datetime(start_time, unit="s")
    time_axis = start_dt + pd.to_timedelta(offsets, unit="s")

    return time_axis

def coalesce_time_pressure(data_list, concatenate=False):
    """
    Coalesce multiple time and pressure arrays into single lists.

    Parameters
    ----------
    data_list : list of dicts
        List of dictionaries, each containing 'start_time', 'times', and 'pressure' keys
    concatenate : bool, optional
        If True, concatenate the lists into single numpy arrays. If False, return as lists of arrays. Default is False.

    Returns
    -------
    coalesced_time : list or np.ndarray
        Coalesced time data, either as a list of arrays or a single concatenated array
    coalesced_pressure : list or np.ndarray
        Coalesced pressure data, either as a list of arrays or a single concatenated array
    """
    coalesced_time = []
    coalesced_pressure = []
    
    for data in data_list:
        time_axis = convert_to_datetime(data['start_time'], data['times'])
        coalesced_time.append(time_axis)
        coalesced_pressure.append(data['pressure'])
    
    if concatenate:
        coalesced_time = np.concatenate(coalesced_time)
        coalesced_pressure = np.concatenate(coalesced_pressure)
    
    return coalesced_time, coalesced_pressure

