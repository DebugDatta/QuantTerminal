"""ML regression models for return/price prediction."""

import numpy as np
from typing import Any


def get_model(name: str, **params) -> Any:
    """Get an ML model by name."""
    if name == "Linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    elif name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    elif name == "Lasso":
        from sklearn.linear_model import Lasso
        return Lasso(**params)
    elif name == "ElasticNet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**params)
    elif name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42, **params)
    elif name == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, random_state=42, **params)
    elif name == "SVR":
        from sklearn.svm import SVR
        return SVR(**params)
    elif name == "KNN":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**params)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=42, **params)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, **params)
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=100, random_seed=42, verbose=False, **params)
    else:
        raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = [
    "Linear", "Ridge", "Lasso", "ElasticNet",
    "Random Forest", "Gradient Boosting", "SVR", "KNN",
    "XGBoost", "LightGBM", "CatBoost",
]
