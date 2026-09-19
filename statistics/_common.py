"""Shared validation and cleaning utilities for statistics modules.

This module provides common helper functions used across multiple statistics
submodules to avoid code duplication and ensure consistent validation behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shared constants
MIN_ASSETS = 2
MIN_ASSET_COLUMNS = 2
MIN_COMPLETE_OBSERVATIONS = 3


def clean_series(series: pd.Series) -> pd.Series:
    """Clean and validate a pandas Series by removing NaN values.

    Parameters
    ----------
    series : pd.Series
        Input series to clean.

    Returns
    -------
    pd.Series
        Cleaned series with NaN values removed.

    Raises
    ------
    TypeError
        If input is not a pandas Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    return series.dropna()


def clean_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a pandas DataFrame by removing rows with NaN.

    Uses listwise deletion (drops rows where any column has NaN) per the
    DATA_LAYER inner-join rule for multi-asset operations.

    Parameters
    ----------
    returns : pd.DataFrame
        Input DataFrame to clean. Rows = observations, columns = assets.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with NaN rows removed.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    return returns.dropna(axis=0, how="any")


def validate_multi_asset(clean: pd.DataFrame, min_assets: int = MIN_ASSETS, min_obs: int = MIN_COMPLETE_OBSERVATIONS, label: str = "correlation/covariance") -> None:
    """Validate that a cleaned DataFrame has sufficient assets and observations.

    Parameters
    ----------
    clean : pd.DataFrame
        Already-cleaned DataFrame to validate.
    min_assets : int, default MIN_ASSETS
        Minimum number of asset columns required.
    min_obs : int, default MIN_COMPLETE_OBSERVATIONS
        Minimum number of complete observations required.
    label : str, default "correlation/covariance"
        Label for error messages.

    Raises
    ------
    ValueError
        If the DataFrame has insufficient assets or observations.
    """
    if clean.shape[1] < min_assets:
        raise ValueError(
            f"{label} requires at least {min_assets} asset columns; "
            f"got {clean.shape[1]}"
        )
    if clean.shape[0] < min_obs:
        raise ValueError(
            f"{label} requires at least {min_obs} complete observations; "
            f"got {clean.shape[0]}"
        )


def validate_bool_as_int(value, name: str) -> None:
    """Validate that a value is not a boolean used as an integer.

    This is important in Streamlit contexts where widget values may be
    coerced to bool.

    Parameters
    ----------
    value : any
        Value to validate.
    name : str
        Name of the parameter for error messages.

    Raises
    ------
    TypeError
        If value is a boolean.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean")
