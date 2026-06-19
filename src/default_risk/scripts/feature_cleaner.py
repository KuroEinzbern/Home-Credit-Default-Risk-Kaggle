import numpy as np
import pandas as pd
from typing import Any
import sklearn as skt
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.base import BaseEstimator
import mlflow.sklearn
from sklearn.model_selection import BaseCrossValidator
from mlflow.models import infer_signature
from default_risk.config import ARTIFACTS_DIR
from functools import reduce
from sklearn.inspection import permutation_importance


def clean_importance_zero_and_negative_pfi( feature_importance_from_pfi: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame :
    feature_importance_from_pfi= feature_importance_from_pfi.sort_values("mean_importance_cv")
    features_to_drop= feature_importance_from_pfi [feature_importance_from_pfi["mean_importance_cv"] <= 0] 
    features_names= features_to_drop["feature"].tolist() 
    print(f"eliminando {features_names} por importancia 0 o negativa en feature permutation")
    return X.drop(columns=features_names)