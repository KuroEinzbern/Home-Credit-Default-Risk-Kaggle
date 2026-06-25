import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from default_risk.config import ARTIFACTS_DIR
from functools import reduce
from sklearn.inspection import permutation_importance


def clean_importance_zero_and_negative_pfi( feature_importance_from_pfi: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame :
    feature_importance_from_pfi= feature_importance_from_pfi.sort_values("mean_importance_cv")
    features_to_drop= feature_importance_from_pfi [feature_importance_from_pfi["mean_importance_cv"] <= 0] 
    features_names= features_to_drop["feature"].tolist() 
    print(f"eliminando {features_names} por importancia 0 o negativa en feature permutation")
    return X.drop(columns=features_names)

def clean_noise_from_feature_importance(feature_importance: pd.DataFrame, X: pd.DataFrame, minimun_importance : float = 0) -> pd.DataFrame :
    features_to_drop= feature_importance [(feature_importance["importances"] < minimun_importance) ] #& (feature_importance["importances"] > 0)
    
    features_names= features_to_drop["feature_name"].tolist() 
    print(f"eliminando {features_names} por importancia debajo del umbral {minimun_importance}")
    return X.drop(columns=features_names)


def creating_criteria(feature_importance: pd.DataFrame, fpi: pd.DataFrame) :
    merged= feature_importance.merge(fpi, how="inner", left_on="feature_name", right_on="feature")
    merged["diff_gain_and_permutation"]= merged["importances"] - merged["mean_importance_cv"] 
    merged.sort_values("diff_gain_and_permutation",ascending=False,inplace=True)
    merged.to_csv(ARTIFACTS_DIR / "criteria.csv")
    return merged