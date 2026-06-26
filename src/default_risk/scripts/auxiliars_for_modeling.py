import pandas as pd
from scipy.stats import trim_mean       
from IPython.display import display
import numpy as np
import default_risk.config as cfg
from typing import Any
from sklearn.model_selection import StratifiedKFold


def cast_object_into_categoricals(X : pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category')
    return X

def prepare_columns(df : pd.DataFrame, columns_to_drop : list=["id_curr"], target_name : str ="target")  -> tuple[ pd.DataFrame, pd.Series] :
    Y= df[target_name]
    X= df.drop(columns=[target_name])
    X.drop(columns=columns_to_drop,inplace=True)
    return X,Y

def get_baseline_setup() -> tuple[StratifiedKFold, dict[str, Any]] :
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    hiperparams=  {  
        "objective" : 'binary:logistic',
        "random_state" : 42,
        "eval_metric" :"auc",
        "enable_categorical" : True,
        "min_child_weight" : 30
    }
    return cv,hiperparams

        