import pandas as pd
import xgboost as xgb
from default_risk.scripts.cv_mlfow_integration import run_cv_tracked_mlflow
import default_risk.config as cfg
import os
import xgboost as xgb
from dotenv import load_dotenv
import joblib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from default_risk.config import CLEANS_DIR, DATA_DIR, PROCESSED_DIR, RAW_DATA_DIR, SPLITS_DIR, CANONIC_DIR
from default_risk.data.clean import clean_bureau, clean_credit_card_balance, clean_installments_payments, clean_pos_cash_balance, clean_previous_application, clean_application_train,clean_bureau_balance
from default_risk.data.extract import download_dataset, split_dataset, canonizate
from collections.abc import Callable



from default_risk.data.process import process_application_train, process_bureau, process_bureau_balance, process_credit_card_balance, process_installments_payments, process_pos_cash_balance, process_previous_application
canonizated_tables: dict ={}

from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns
from default_risk.scripts.auxiliars_for_modeling import get_pipeline
from default_risk.data.clean import get_cleaning_dict
from default_risk.data.make_dataset import apply_cleaning



def main():
    pipeline= load_model(cfg.MODELS_DIR / "model-1.1.pkl")
    test_application= pd.read_parquet(cfg.MASTER_DATA_DIR / "prepared_dataset_test.parquet")
    id_to_predict= test_application["id_curr"]
    X_test= test_application.drop(columns= ["id_curr"])
    X_test = cast_object_into_categoricals(X_test)
    probas = pipeline.predict_proba(X_test)
    default_predictions= probas[:, 1]
    submision= pd.DataFrame({"SK_ID_CURR" : id_to_predict, "TARGET": default_predictions})
    submision.to_csv(cfg.MASTER_DATA_DIR / "submision_kaggle.csv",index=False)
        


def build_data(df_aplications_to_predict : pd.DataFrame):
    ids_to_predict= df_aplications_to_predict["SK_ID_CURR"]
    cleaning_dict= get_cleaning_dict()
    for table_name in cleaning_dict.keys() :
        apply_cleaning(table_name,cleaning_dict,True,ids_to_predict)
    return

def load_model(input_path : str):
     return joblib.load(input_path)
   

def inference(df_data_to_predict: pd.DataFrame):
    return


if __name__ == "__main__": main()