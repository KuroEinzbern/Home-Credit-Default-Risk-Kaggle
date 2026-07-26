
import pandas as pd
import xgboost as xgb
import yaml
from default_risk.scripts.cv_mlfow_integration import run_cv_tracked_mlflow
import default_risk.config as cfg
import os
import xgboost as xgb
from dotenv import load_dotenv
import joblib

from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns
from default_risk.scripts.auxiliars_for_modeling import get_pipeline

def load_xgboost_params() -> dict:
    
    if not cfg.MODEL_PARAMS.exists():
        raise FileNotFoundError(
            f"There is are no hyperparams defined in {cfg.MODEL_PARAMS}"
        )
        
    with open(cfg.MODEL_PARAMS, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Extrae el diccionario bajo la clave "xgboost"
    return config.get("xgboost", {})


def main():
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv()
    
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")
    cv, _ = get_baseline_setup()
    dataset = pd.read_parquet(cfg.MASTER_DATA_DIR / 'prepared_dataset_train.parquet')
    X,Y = prepare_columns(dataset)
    X = cast_object_into_categoricals(X)

    hiperparams = load_xgboost_params()
    features_for_target_encoding= []

    model= xgb.XGBClassifier(**hiperparams)
    pipeline= get_pipeline(50,features_for_target_encoding, model)

    run_cv_tracked_mlflow(pipeline,hiperparams,cv,X,Y,experiment_name,"Pipeline-1.0")

    pipeline.fit(X,Y)

    joblib.dump(pipeline, cfg.MODELS_DIR /'model-1.1.pkl')

if __name__ == "__main__": main()