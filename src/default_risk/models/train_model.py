
import pandas as pd
import gc
import xgboost as xgb
from default_risk.scripts.cv_mlfow_integration import run_cv_tracked_mlflow
import default_risk.config as cfg
import os
import xgboost as xgb
from dotenv import load_dotenv

from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns


def main():
    load_dotenv()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")
    cv,hiperparams = get_baseline_setup()
    dataset = pd.read_parquet(cfg.MASTER_DATA_DIR / 'prepared_dataset.parquet')
    X,Y = prepare_columns(dataset)

    X = cast_object_into_categoricals(X)
    run_cv_tracked_mlflow(xgb.XGBClassifier,hiperparams,cv,X,Y,experiment_name,"Pipeline-1.0")

if __name__ == "__main__": main()