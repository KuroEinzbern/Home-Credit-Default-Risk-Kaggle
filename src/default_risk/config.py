from pathlib import Path

# __file__ es la posición de este script. 
# .parent.parent.parent sube desde 'config.py' hasta la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Definición de carpetas principales
DATA_DIR = PROJECT_ROOT / "data/"
NOTEBOOKS= PROJECT_ROOT / "notebooks"
METADATA_DIR = PROJECT_ROOT / "metadata"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"


RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MASTER_DATA_DIR = DATA_DIR / "master"

CANONIC_DIR = INTERIM_DATA_DIR / "00_Canonicals"
SPLITS_DIR = INTERIM_DATA_DIR / "01_Splits"
CLEANS_DIR = INTERIM_DATA_DIR / "02_Cleans"
PROCESSED_DIR = INTERIM_DATA_DIR / "03_Processed"

MODEL_PARAMS = PROJECT_ROOT / "params.yaml"
DUMP_FROM_NOTEBOOKS= NOTEBOOKS / "dumps_from_notebooks"

# Rutas específicas a archivos
APPLICATION_TRAIN = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST = RAW_DATA_DIR / "application_test.csv"
BUREAU = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE = RAW_DATA_DIR / "bureau_balance.csv"
CREDIT_CARD_BALANCE= RAW_DATA_DIR / "credit_card_balance.csv"
INSTALLMENTS_PAYMENTS= RAW_DATA_DIR / "installments_payments.csv"
POS_CASH_BALANCE = RAW_DATA_DIR / "POS_CASH_balance.csv"
PREVIOUS_APPLICATION= RAW_DATA_DIR / "previous_application.csv"
SCHEMA_JSON = METADATA_DIR / "schema.json"

# Crear carpetas si no existen
DATA_DIR.mkdir(exist_ok=True)
DUMP_FROM_NOTEBOOKS.mkdir(exist_ok=True)
CANONIC_DIR.mkdir(exist_ok=True,parents=True)
SPLITS_DIR.mkdir(exist_ok=True,parents=True)
CLEANS_DIR.mkdir(exist_ok=True,parents=True)
INTERIM_DATA_DIR.mkdir(exist_ok=True,parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True,parents=True)
PROCESSED_DIR.mkdir(exist_ok=True,parents=True)




feature_filter_for_xgb= ['bureau_balance_is_delincuency_sum_loan_1', 'closed_days_credit_update_closed_max', 'instalments_completion_ratio_mean', 'last_365_instalments_days_of_delinquency_mean', 'implied_interest_rate_mean', 'last_365_instalments_is_delinquency_mean', 'active_amt_credit_sum_active_max', 'ratio_credit_to_goods_max', 'ext_source_2', 'active_completetitud_ratio_active_min', 'closed_amt_credit_sum_debt_closed_mean', 'payment_trend', 'active_id_curr_active_count', 'wallsmaterial_mode', 'flag_document_3', 'code_reject_reason_prev_1', 'ext_2_x_3', 'bureau_balance_status_score_mean_loan_1', 'active_amt_annuity_active_std', 'ext_source_3', 'region_raiting_client_city', 'amt_goods_price_max', 'amt_down_payment_sum', 'days_and_insurance_information_are_missing_mean', 'education_type_Secondary / secondary special', 'amt_credit', 'closed_days_credit_update_closed_min', 'name_income_type', 'days_termination_prev_1', 'building_score_mean', 'closed_ratio_credit_annuity_closed_max', 'code_gender', 'bureau_days_enddate_fact_loan_1', 'bureau_credit_type_loan_1', 'credit_duration', 'implied_interest_rate_max', 'last_6_cash_balance_count_instalment_future_min', 'occupation_type', 'last_6_credit_card_balance_limit_ratio_mean', 'bureau_balance_is_delincuency_mean_loan_1', 'bureau_amt_credit_max_overdue_loan_1', 'amt_annuity', 'implied_interest_rate_std', 'closed_amt_credit_sum_debt_closed_std', 'ext_source_1', 'ratio_credit_to_goods_mean', 'organization_type_Bank', 'last_6_credit_card_balance_limit_ratio_max', 'organization_type_Construction', 'ratio_days_employed_days_lived', 'last_6_cash_balance_sk_dpd_def_sum', 'amt_req_credit_breau_day', 'active_amt_credit_max_overdue_active_max', 'instalments_days_of_delinquency_max_max', 'active_have_amt_credit_sum_overdue_active_sum', 'bureau_amt_credit_sum_debt_is_missing_loan_1', 'days_birth', 'days_decision_mean', 'days_decision_min', 'closed_amt_credit_sum_closed_sum', 'last_6_cash_balance_sk_dpd_mean', 'def_60_cnt_social_circle', 'last_365_instalments_amt_payment_min', 'instalments_amt_instalment_sum_sum', 'active_completetitud_ratio_active_mean', 'own_car_age', 'last_6_cash_balance_amount_advanced_payment_max', 'def_30_cnt_social_circle', 'flag_own_realty', 'education_type_Higher education', 'ratio_debt_age', 'documents_count', 'family_status', 'applications_count', 'building_score_sum', 'active_amt_annuity_is_missing_active_sum', 'region_raiting_client', 'amt_req_credit_breau_qrt', 'ext_1_x_3', 'organization_type_Transport: type 3', 'diff_application_credit_max', 'active_days_credit_enddate_active_mean', 'instalments_is_delinquency_mean_mean', 'instalments_extra_instalament_mean_mean', 'ext_1_x_2', 'active_amt_credit_max_overdue_active_sum', 'last_6_credit_card_completion_ratio', 'name_contract_type', 'last_365_instalments_extra_instalament_mean', 'credit_card_is_over_the_limit_mean_mean', 'ext_source_1_is_missing', 'last_6_cash_balance_count_instalment_future_max', 'ratio_good_credit', 'closed_amt_credit_sum_closed_max', 'last_365_instalments_completion_ratio', 'instalments_amt_payment_sum_sum', 'days_id_publish', 'organization_type_Self-employed', 'credit_card_cnt_drawings_atm_current_mean_mean', 'instalments_days_of_delinquency_mean_prev_1', 'name_seller_industry_prev_1', 'last_90_instalments_completion_ratio', 'ext_source_mean', 'organization_type_Military', 'bureau_days_credit_enddate_loan_1', 'flag_not_live_city', 'instalments_days_of_underpayment_max_max', 'kui_ratio', 'closed_amt_credit_max_overdue_closed_mean', 'days_employed', 'instalments_amount_of_versions_in_sequence_prev_1', 'days_last_due_1st_version_prev_1', 'bureau_completetitud_ratio_loan_1', 'amt_goods_price', 'instalments_days_of_delinquency_mean_mean']

