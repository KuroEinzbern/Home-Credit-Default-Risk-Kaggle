import pandas as pd
from scipy.stats import trim_mean       
from IPython.display import display
import numpy as np
import default_risk.config as cfg
from typing import Any
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder


def cast_object_into_categoricals(X : pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category')
    return X

def apply_cyclical_encoding(df, column, max_val) -> pd.DataFrame:
    
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)

    return df

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
        "min_child_weight" : 30,
        "n_jobs": -1,
        "colsample_bytree" : 0.8
    }
    return cv,hiperparams

def get_pipeline(smoothering_param, features_for_target_encoding,model)-> Pipeline:
    preprocessor = ColumnTransformer(
    transformers=[
        ('target_encode_cat', TargetEncoder(
                categories='auto',      
                smooth= smoothering_param,          
                cv=5,                   
                random_state=42
            ), 
            features_for_target_encoding
        )
    ],
    remainder='passthrough' 
)

    preprocessor.set_output(transform="pandas") #god bless this
    pipeline = Pipeline([ ('preprocessor', preprocessor),('classifier', model)])
    return pipeline

        