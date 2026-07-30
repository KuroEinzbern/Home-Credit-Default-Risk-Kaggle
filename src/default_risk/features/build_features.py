from default_risk.config import MASTER_DATA_DIR
import pandas as pd
import gc
import xgboost as xgb
from default_risk.scripts.cv_mlfow_integration import run_cv_tracked_mlflow
import default_risk.config as cfg
import os
import numpy as np  
import xgboost as xgb
from dotenv import load_dotenv

from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns


def main():
    MASTER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits = ['test',"train"]


    for split in splits :
        #app_train_with_feature_engineering + prev_app + bureau + installments
        application_train_df = pd.read_parquet(cfg.PROCESSED_DIR / f"application_{split}-processed.parquet")
        bureau_df = pd.read_parquet(cfg.PROCESSED_DIR / f"bureau_{split}-processed.parquet")

        merged_df = application_train_df.merge(
            bureau_df, 
            on="id_curr", 
            how="left"
        )

        #cleaning the first 2 df before load the third one to avoid RAM bottleneck
        del application_train_df , bureau_df
        gc.collect()

        previous_application_df=  pd.read_parquet(cfg.PROCESSED_DIR / f"previous_application_{split}-processed.parquet")


        merged_df = merged_df.merge(
            previous_application_df,
            on= "id_curr", 
            how="left"
        )

        #cleaning the third
        del previous_application_df
        gc.collect()

        merged_df["instalment_income_ratio"] = np.where(merged_df["amt_income_total"],merged_df["instalments_amt_instalment_sum_sum"] / merged_df["amt_income_total"],np.nan)

        merged_df["avg_annuity_vs_actual_annuity"] = np.where(merged_df["amt_annuity"],merged_df["amt_annuity_median"] / merged_df["amt_annuity"],np.nan)


        
      
        merged_df.to_parquet(MASTER_DATA_DIR / f'prepared_dataset_{split}.parquet')


if __name__ == "__main__": main()
    