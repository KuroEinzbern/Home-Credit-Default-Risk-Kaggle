from default_risk.config import MASTER_DATA_DIR
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

        internal_list = [
        "applications_count",
        "instalments_amount_of_versions_in_sequence_prev_1",
        "instalments_days_of_delinquency_max_max",
        "name_contract_type",
        "last_365_instalments_completion_ratio",
        "days_termination_prev_1",
        "payment_trend",
        "amt_goods_price_max",
        "days_last_due_1st_version_prev_1",
        "organization_type_Construction",
        "ratio_credit_to_goods_max",
        "organization_type_Self-employed",
        "diff_application_credit_max",
        "implied_interest_rate_std",
        "last_6_cash_balance_amount_advanced_payment_max",
        "last_6_credit_card_balance_limit_ratio_mean",
        "amt_req_credit_breau_qrt",
        "building_score_sum",
        "name_seller_industry_prev_1",
        "days_decision_min",
        "implied_interest_rate_mean",
        "def_60_cnt_social_circle",
        "days_decision_mean",
        "ratio_debt_age",
        "instalments_days_of_underpayment_max_max",
        "last_365_instalments_amt_payment_min",
        "instalments_amt_instalment_sum_sum",
        "last_90_instalments_completion_ratio",
        "amt_annuity",
        "instalments_completion_ratio_mean",
        "last_6_cash_balance_count_instalment_future_min",
        "days_employed",
        "amt_credit",
        "instalments_days_of_delinquency_mean_mean",
        "ratio_credit_to_goods_mean",
        "family_status",
        "days_birth",
        "ext_source_3",
        "days_and_insurance_information_are_missing_mean",
        "instalments_days_of_delinquency_mean_prev_1",
        "credit_card_is_over_the_limit_mean_mean",
        "ext_source_1",
        "ext_1_x_2",
        "kui_ratio",
        "credit_card_cnt_drawings_atm_current_mean_mean",
        "last_6_cash_balance_count_instalment_future_max",
        "last_365_instalments_is_delinquency_mean",
        "amt_down_payment_sum",
        "name_income_type",
        "documents_count",
        "amt_req_credit_breau_day",
        "amt_goods_price",
        "region_raiting_client_city",
        "last_365_instalments_extra_instalament_mean",
        "last_6_credit_card_completion_ratio",
        "flag_not_live_city",
        "instalments_is_delinquency_mean_mean",
        "own_car_age",
        "code_reject_reason_prev_1",
        "def_30_cnt_social_circle",
        "instalments_amt_payment_sum_sum",
        "instalments_extra_instalament_mean_mean",
        "occupation_type",
        "credit_duration",
        "implied_interest_rate_max",
        "last_365_instalments_days_of_delinquency_mean",
        "last_6_cash_balance_sk_dpd_def_sum",
        "education_type_Secondary / secondary special",
        "ratio_good_credit",
        "region_raiting_client",
        "code_gender",
        "last_6_cash_balance_sk_dpd_mean",
        "flag_document_3",
        "last_6_credit_card_balance_limit_ratio_max",
        "ext_2_x_3",
        "education_type_Higher education",
        "ext_source_mean"
        ]


        external_list= [
        "ext_source_1_is_missing",
        "amt_req_credit_breau_day",
        "organization_type_Bank",
        "region_raiting_client",
        "closed_ratio_credit_annuity_closed_max",
        "organization_type_Construction",
        "active_amt_annuity_active_std",
        "bureau_days_enddate_fact_loan_1",
        "active_have_amt_credit_sum_overdue_active_sum",
        "flag_own_realty",
        "closed_days_credit_update_closed_max",
        "building_score_sum",
        "closed_amt_credit_sum_debt_closed_mean",
        "active_amt_credit_max_overdue_active_max",
        "closed_days_credit_update_closed_min",
        "bureau_balance_is_delincuency_mean_loan_1",
        "active_amt_annuity_is_missing_active_sum",
        "active_days_credit_enddate_active_mean",
        "active_amt_credit_sum_active_max",
        "closed_amt_credit_sum_debt_closed_std",
        "bureau_days_credit_enddate_loan_1",
        "closed_amt_credit_sum_closed_max",
        "organization_type_Military",
        "building_score_mean",
        "amt_req_credit_breau_qrt",
        "bureau_balance_is_delincuency_sum_loan_1",
        "days_id_publish",
        "ratio_debt_age",
        "wallsmaterial_mode",
        "closed_amt_credit_max_overdue_closed_mean",
        "bureau_balance_status_score_mean_loan_1",
        "ratio_days_employed_days_lived",
        "organization_type_Self-employed",
        "closed_amt_credit_sum_closed_sum",
        "kui_ratio",
        "ext_1_x_3",
        "organization_type_Transport: type 3",
        "active_amt_credit_max_overdue_active_sum",
        "days_birth",
        "amt_credit",
        "def_60_cnt_social_circle",
        "bureau_amt_credit_max_overdue_loan_1",
        "name_contract_type",
        "ext_source_1",
        "name_income_type",
        "amt_annuity",
        "active_id_curr_active_count",
        "days_employed",
        "bureau_amt_credit_sum_debt_is_missing_loan_1",
        "amt_goods_price",
        "family_status",
        "bureau_completetitud_ratio_loan_1",
        "active_completetitud_ratio_active_mean",
        "occupation_type",
        "region_raiting_client_city",
        "bureau_credit_type_loan_1",
        "ext_2_x_3",
        "ext_1_x_2",
        "own_car_age",
        "flag_not_live_city",
        "active_completetitud_ratio_active_min",
        "credit_duration",
        "education_type_Secondary / secondary special",
        "def_30_cnt_social_circle",
        "documents_count",
        "flag_document_3",
        "code_gender",
        "ratio_good_credit",
        "ext_source_3",
        "ext_source_2",
        "education_type_Higher education",
        "ext_source_mean"
        ]

        extra_features = []
        extra_features.append('id_curr')
        if(split == "train"):
            extra_features.append('target')
        features_names = list(set(internal_list + external_list + extra_features))
        final_dataset = merged_df[features_names]
        final_dataset.to_parquet(MASTER_DATA_DIR / f'prepared_dataset_{split}.parquet')


if __name__ == "__main__": main()
    