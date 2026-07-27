import pickle
import pandas as pd
import numpy as np
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
        coalesced_pressure.append(data['pressure'] + data['ref_pressure'])
    
    if concatenate:
        coalesced_time = np.concatenate(coalesced_time)
        coalesced_pressure = np.concatenate(coalesced_pressure)
    
    return coalesced_time, coalesced_pressure

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


MET4A_SAMPLE_DTYPE = np.dtype(
    [
        ("time", "f8"),
        ("pressure", "f8"),
        ("temp", "f8"),
        ("rh", "f8"),
        ("ref_pressure", "f8"),
        ("scale_fac", "f8"),
    ]
)


def load_met4a_pickle(path: str | Path) -> dict:
    """
    Load a dictionary from a MET4A pickle file.
    """
    with Path(path).open("rb") as file:
        data = pickle.load(file)
    return data


def met4a_dict_to_recarray(
    data: dict,
) -> np.recarray:
    """
    Convert one MET4A dictionary into a sample record array.
    """

    samples = np.empty(int(data["nsamp"]), dtype=MET4A_SAMPLE_DTYPE)

    samples["time"] = float(data["start_time"]) + np.asarray(
        data["times"], dtype=np.float64
    )

    for field in ("pressure", "temp", "rh"):
        samples[field] = np.asarray(data[field], dtype=np.float64)

    samples["ref_pressure"] = float(data["ref_pressure"])
    samples["scale_fac"] = float(data["scale_fac"])

    return samples.view(np.recarray)


def concatenate_met4a_recarrays(
    blocks: Iterable[np.ndarray],
    *,
    sort: bool = True,
) -> np.recarray:
    """
    Combine MET4A arrays into one chronological recarray.
    """
    arrays = [np.asarray(block) for block in blocks]

    if not arrays:
        return np.empty(0, dtype=MET4A_SAMPLE_DTYPE).view(np.recarray)

    if sort:
        arrays.sort(
            key=lambda array: (
                float(array["time"][0]) if len(array) else float("inf")
            )
        )

    samples = np.concatenate(arrays).view(np.recarray)
    if sort and len(samples):
        order = np.argsort(samples.time, kind="stable")
        samples = samples[order].view(np.recarray)

    return samples


def load_met4a_pickles(
    paths_or_directory: str | Path | Iterable[str | Path],
    *,
    pattern: str = "*.met4a.pkl",
) -> np.recarray:
    """
    Load pickle blocks into one timestamp-sorted recarray.
    """
    if isinstance(paths_or_directory, (str, Path)):
        candidate = Path(paths_or_directory)
        paths = list(candidate.glob(pattern)) if candidate.is_dir() else [candidate]
    else:
        paths = [Path(path) for path in paths_or_directory]

    blocks = [
        met4a_dict_to_recarray(load_met4a_pickle(path))
        for path in paths
    ]
    return concatenate_met4a_recarrays(blocks)


def find_time_gaps(
    samples: np.ndarray,
    *,
    gap_factor: float = 2.0,
) -> np.recarray:
    """
    Report intervals larger than ``gap_factor`` times the median period.
    """
    gap_dtype = np.dtype(
        [
            ("index_before", "i8"),
            ("start_time", "f8"),
            ("end_time", "f8"),
            ("duration", "f8"),
            ("estimated_missing", "i8"),
        ]
    )
    times = np.asarray(samples["time"], dtype=np.float64)
    if len(times) < 2:
        return np.empty(0, dtype=gap_dtype).view(np.recarray)

    deltas = np.diff(times)
    sample_period = float(np.median(deltas))
    indices = np.flatnonzero(deltas > gap_factor * sample_period)
    gaps = np.empty(len(indices), dtype=gap_dtype)
    gaps["index_before"] = indices
    gaps["start_time"] = times[indices]
    gaps["end_time"] = times[indices + 1]
    gaps["duration"] = deltas[indices]
    gaps["estimated_missing"] = np.maximum(
        np.rint(deltas[indices] / sample_period).astype(np.int64) - 1,
        0,
    )
    return gaps.view(np.recarray)