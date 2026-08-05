import pandas as pd
import xgboost as xgb
from default_risk.scripts.cv_mlfow_integration import run_cv_tracked_mlflow
import default_risk.config as cfg
import os
import xgboost as xgb
from dotenv import load_dotenv
import argparse
import joblib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from default_risk.config import CLEANS_DIR, DATA_DIR, PROCESSED_DIR, RAW_DATA_DIR, SPLITS_DIR, CANONIC_DIR
from default_risk.data.clean import clean_bureau, clean_credit_card_balance, clean_installments_payments, clean_pos_cash_balance, clean_previous_application, clean_application_train,clean_bureau_balance
from default_risk.data.extract import download_dataset, split_dataset, canonizate
from collections.abc import Callable
import re



from default_risk.data.process import process_application_train, process_bureau, process_bureau_balance, process_credit_card_balance, process_installments_payments, process_pos_cash_balance, process_previous_application
canonizated_tables: dict ={}

from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns
from default_risk.scripts.auxiliars_for_modeling import get_pipeline
from default_risk.data.clean import get_cleaning_dict
from default_risk.data.make_dataset import apply_cleaning



def main(args):
    test_application= pd.read_parquet(cfg.MASTER_DATA_DIR / "prepared_dataset_test.parquet")
    id_to_predict= test_application["id_curr"]


    if(args.only_xgb) :
        default_proba= predict_with("xgb",test_application)
        create_csv_with_predictions(id_to_predict,default_proba)
        return
    
    if(args.only_lgbm) :
        default_proba= predict_with("lgbm",test_application)
        create_csv_with_predictions(id_to_predict,default_proba)
        return
    
    if((args.all_models) or (any(vars(args).values()))) :
        default_proba_xgb= predict_with("xgb",test_application)
        default_proba_lgbm= predict_with("lgbm",test_application)
        prediction_mean= (default_proba_xgb *0.2679) + (default_proba_lgbm * 0.7321)
        create_csv_with_predictions(id_to_predict,prediction_mean)
    


def predict_with(model_name : str, applications_to_predict : pd.DataFrame):
    pipeline= load_model(cfg.MODELS_DIR / f"model-{model_name}-1.1.pkl")
    X_test= applications_to_predict.drop(columns= ["id_curr"])
    X_test= X_test.rename(columns=lambda c: re.sub(r'[^A-Za-z0-9_]', '_', c))
    X_test = cast_object_into_categoricals(X_test)
    probas = pipeline.predict_proba(X_test)
    return probas[:, 1]



def create_csv_with_predictions(id_to_predict, default_proba):
    submision= pd.DataFrame({"SK_ID_CURR" : id_to_predict, "TARGET": default_proba})
    submision.to_csv(cfg.MASTER_DATA_DIR / "submision_kaggle.csv",index=False)
    return
    
        


def load_model(input_path : str):
     return joblib.load(input_path)
   


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_xgb", help="only train xgb")
    parser.add_argument("--only_lgbm", help="only train lgbm")
    parser.add_argument("--all_models", help="prediction is a mean of the predictions of all the models")
    args = parser.parse_args()
    main(args)