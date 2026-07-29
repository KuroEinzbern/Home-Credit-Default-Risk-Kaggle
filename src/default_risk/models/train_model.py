
import pandas as pd
import xgboost as xgb
import yaml
from default_risk.scripts.cv_mlfow_integration import run_cv_tracked_mlflow
import default_risk.config as cfg
import os
import xgboost as xgb
from dotenv import load_dotenv
import joblib
import lightgbm as lgb
from sklearn.base import BaseEstimator
from functools import singledispatch
from typing import Any
import argparse
from collections.abc import Callable



from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns
from default_risk.scripts.auxiliars_for_modeling import get_pipeline


def load_params(param_name)-> dict:
    if not cfg.MODEL_PARAMS.exists():
        raise FileNotFoundError(
            f"There is are no hyperparams defined in {cfg.MODEL_PARAMS}"
        )
        
    with open(cfg.MODEL_PARAMS, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config.get(param_name, {})
    

def instance_xgb() -> tuple[BaseEstimator,  dict[str, Any]]: 
    hiperparams = load_params("xgboost")
    print(hiperparams)
    model= xgb.XGBClassifier(**hiperparams)
    return model , hiperparams



def instance_lgbm() -> tuple[BaseEstimator,  dict[str, Any]]: 
    hiperparams = load_params("lgbm")
    model= lgb.LGBMClassifier(**hiperparams)
    return model , hiperparams 


procesing_dict: dict[str, Callable] = {
    "lgbm": instance_xgb,
    "xgb": instance_lgbm,
}



def main(args):

    if(args.only_xgb) :
        train_model("xgb")
        return
    if(args.only_lgbm) :
        train_model("lgbm")
        return

    #without specification, we train both
    train_model("xgb")
    train_model("lgbm")


def train_model(model_class) :

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")
    cv, _ = get_baseline_setup()


    dataset = pd.read_parquet(cfg.MASTER_DATA_DIR / 'prepared_dataset_train.parquet')

    if(model_class == "lgbm") : sanitize= True
    if(model_class == "xgb") : sanitize= False

    X,Y = prepare_columns(dataset,santize_text=sanitize)
    X = cast_object_into_categoricals(X)
    categorical_features= ["organization_type","occupation_type"]


    factory = procesing_dict[model_class]
    instanced_model , hiperparams = factory()
    pipeline= get_pipeline(50,categorical_features, instanced_model)


    features = hiperparams["features"]
    X= X[features]


    run_cv_tracked_mlflow(pipeline,hiperparams,cv,X,Y,experiment_name,"Pipeline-1.0")
    pipeline.fit(X,Y)
    joblib.dump(pipeline, cfg.MODELS_DIR /f'model-{model_class.__name__}-1.1.pkl')
    return






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_xgb", help="only train xgb")
    parser.add_argument("--only_lgbm", help="only train lgbm")

    args = parser.parse_args()
    main(args)