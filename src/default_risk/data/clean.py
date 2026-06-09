import pandas as pd
import numpy as np
from pathlib import Path

def clip_p999(series, set_negatives_to_nan=False):
    s = series.copy()
    if set_negatives_to_nan:
        s.loc[s < 0] = np.nan
    p99_9 = s.quantile(0.999)
    s.loc[s > p99_9] = p99_9
    return s


def clip_p99_x4_and_fill(series, fill_nulls=False):
    s = series.copy()
    p99_x4 = s.quantile(0.99) * 4
    s.loc[s > p99_x4] = p99_x4
    if fill_nulls:
        s = s.fillna(0)
    return s

def agroup_ultra_rare_categories(col : pd.Series, umbral_of_observations : int):
    count = col.value_counts()
    agrouped_col= col.where(col.map(count) < umbral_of_observations , "other")
    return agrouped_col.astype("category")
   
def clean_credit_card_balance(input_filepath: Path, output_filepath: Path):

    print('Starting credit_card_balance table cleaning...')
    print(f'Loading data from: {input_filepath}')
    np.seterr(all='raise')

    df = pd.read_parquet(input_filepath)
    df_clean = pd.DataFrame()

    # id_prev
    df_clean['id_prev'] = df['SK_ID_PREV']
    
    # id_curr
    df_clean['id_curr'] = df['SK_ID_CURR']
    
    # months_balance
    df_clean['months_balance'] = df['MONTHS_BALANCE']

    # have_negative_balance
    df_clean['have_negative_balance'] = (df['AMT_BALANCE'] < 0).astype(int)
    
    # amt_positive_balance
    df_clean['amt_positive_balance'] = df['AMT_BALANCE'].clip(lower=0)

    # amt_negative_balance
    df_clean['amt_negative_balance'] = df['AMT_BALANCE'].clip(upper=0).abs()
    
    # diff_total_receivable_balance
    diff_total_balance = df['AMT_TOTAL_RECEIVABLE'] - df['AMT_BALANCE']
    df_clean['diff_total_receivable_balance'] = np.sign(diff_total_balance) * np.abs(diff_total_balance)

    # amt_credit_limit_actual
    df_clean['amt_credit_limit_actual'] = df['AMT_CREDIT_LIMIT_ACTUAL']

    # amt_drawings_current
    amt_drawings_curr = df['AMT_DRAWINGS_CURRENT'].copy()
    amt_drawings_curr.loc[amt_drawings_curr < 0] = amt_drawings_curr.median()
    df_clean['amt_drawings_current'] = amt_drawings_curr

    # amt_drawings_amt_current
    df['AMT_DRAWINGS_ATM_CURRENT'] = df['AMT_DRAWINGS_ATM_CURRENT'].fillna(0)
    df_clean['amt_drawings_amt_current'] = clip_p999(df['AMT_DRAWINGS_ATM_CURRENT'], set_negatives_to_nan=True)
    
    # amt_drawings_other_current
    df['AMT_DRAWINGS_OTHER_CURRENT'] = df['AMT_DRAWINGS_OTHER_CURRENT'].fillna(0)
    df_clean['amt_drawings_other_current'] = clip_p999(df['AMT_DRAWINGS_OTHER_CURRENT'])
    
    # amt_drawings_pos_current
    df['AMT_DRAWINGS_POS_CURRENT'] = df['AMT_DRAWINGS_POS_CURRENT'].fillna(0)
    df_clean['amt_drawings_pos_current'] = clip_p999(df['AMT_DRAWINGS_POS_CURRENT'])

    # amt_drawings_is_missing
    df_clean['amt_drawings_is_missing'] = df[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].isnull().any(axis=1).astype(int)

    # amt_inst_min_regularity
    df_clean['amt_inst_min_regularity'] = clip_p999(df['AMT_INST_MIN_REGULARITY'])
    
    # amt_inst_min_regularity_is_missing
    df_clean['amt_inst_min_regularity_is_missing'] = df['AMT_INST_MIN_REGULARITY'].isnull().astype(int)

    # REVISAR - QA
    # first_expected_payment_month
    temp_sort = df[['SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY']].sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
    first_months = temp_sort[temp_sort['AMT_INST_MIN_REGULARITY'] > 0].groupby('SK_ID_PREV')['MONTHS_BALANCE'].first()
    df_clean['first_expected_payment_month'] = df['SK_ID_PREV'].map(first_months)

    # inconsistency_gap
    gap_condition = (df['AMT_INST_MIN_REGULARITY'] > df['AMT_TOTAL_RECEIVABLE']) & (df['AMT_TOTAL_RECEIVABLE'] > 0)
    gap_values = df['AMT_INST_MIN_REGULARITY'] - df['AMT_TOTAL_RECEIVABLE']
    df_clean['inconsistency_gap'] = np.where(gap_condition, gap_values.clip(lower=0), 0)

    # diff_payment_current_total
    df_clean['diff_payment_current_total'] = df['AMT_PAYMENT_CURRENT'] - df['AMT_PAYMENT_TOTAL_CURRENT']+1

    # amt_payment_total_current
    df_clean['amt_payment_total_current'] = clip_p999(df['AMT_PAYMENT_TOTAL_CURRENT'])
    
    # amt_payment_total_current_is_missing
    df_clean['amt_payment_total_current_is_missing'] = df['AMT_PAYMENT_TOTAL_CURRENT'].isnull().astype(int)

    # amt_recivable_principal
    s_principal_clipped = np.where(df['AMT_RECEIVABLE_PRINCIPAL'] < 0, 0, df['AMT_RECEIVABLE_PRINCIPAL'])
    df_clean['amt_recivable_principal'] = s_principal_clipped

    # amt_reciavable_principal_positive_balance
    df_clean['amt_reciavable_principal_positive_balance'] = np.where(df['AMT_RECEIVABLE_PRINCIPAL'] < 0, df['AMT_RECEIVABLE_PRINCIPAL'] * -1, 0)

    # amt_recivable
    s_recivable_clipped = np.where(df['AMT_RECIVABLE'] < 0, 0, df['AMT_RECIVABLE'])
    df_clean['amt_recivable'] = s_recivable_clipped

    # amt_recivable_positive_balance
    df_clean['amt_recivable_positive_balance'] = np.where(df['AMT_RECIVABLE'] < 0, df['AMT_RECIVABLE'] * -1, 0)
    
    #amt_total_recivable_positive_balance

    # cnt_drawings_current
    df_clean['cnt_drawings_current'] = clip_p99_x4_and_fill(df['CNT_DRAWINGS_CURRENT'], fill_nulls=False)
    
    # cnt_drawings_amt_current
    df_clean['cnt_drawings_amt_current'] = clip_p99_x4_and_fill(df['CNT_DRAWINGS_ATM_CURRENT'], fill_nulls=True)

    # cnt_drawings_other_current
    df_clean['cnt_drawings_other_current'] = clip_p99_x4_and_fill(df['CNT_DRAWINGS_OTHER_CURRENT'], fill_nulls=True)

    # cnt_drawings_pos_current
    df_clean['cnt_drawings_pos_current'] = clip_p99_x4_and_fill(df['CNT_DRAWINGS_POS_CURRENT'], fill_nulls=True)
        
    # cnt_drawings_are_present
    df_clean['cnt_drawings_are_present'] = df[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT']].isnull().any(axis=1).astype(int)

    # cnt_instalment_mature_cum
    df_clean['cnt_instalment_mature_cum'] = df['CNT_INSTALMENT_MATURE_CUM'].fillna(0)

    # QA
    # name_contract_status
    contract_status = df['NAME_CONTRACT_STATUS'].copy()
    contract_status = contract_status.replace('Sent proposal', 'Signed')
    contract_status = contract_status.replace('Approved', 'Signed')
    contract_status = contract_status.replace('Refused', 'Active')
    df_clean['name_contract_status'] = contract_status

    # sk_dpd
    df_clean['sk_dpd'] = df['SK_DPD']
    
    # sk_dpd_tecnical
    df_clean['sk_dpd_tecnical'] = (df['SK_DPD'] == 1).astype(int)
    
    # sk_dpd_severe
    df_clean['sk_dpd_severe'] = (df['SK_DPD'] > 1).astype(int)

    # sk_dpd_def
    df_clean['sk_dpd_def'] = df['SK_DPD_DEF']
    
    # sk_dpd_def_tecnical
    df_clean['sk_dpd_def_tecnical'] = (df['SK_DPD_DEF'] == 1).astype(int)
    
    # sk_dpd_def_severe
    df_clean['sk_dpd_def_severe'] = (df['SK_DPD_DEF'] > 1).astype(int)


    have_at_least_one_status_closed= df["NAME_CONTRACT_STATUS"].eq("Completed").groupby(df["SK_ID_PREV"]).transform("any")
    have_recent_balance= df["MONTHS_BALANCE"].gt(-4).groupby(df["SK_ID_PREV"]).transform("any")

    is_closed = df["NAME_CONTRACT_STATUS"] == "Completed"

    df_clean["closing_month"] = (df["MONTHS_BALANCE"].where(is_closed).groupby(df["SK_ID_PREV"]).transform("min"))
    df_clean["non_closed_loan"] = (~have_at_least_one_status_closed)
    df_clean["potential_on_going_loan"] = (~have_at_least_one_status_closed) & (have_recent_balance)
    df_clean["incomplete_sequence"] =  (~have_at_least_one_status_closed) & (~have_recent_balance)
    
    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')

def clean_installments_payments(input_filepath: Path, output_filepath: Path):
    print('Starting installments_payments table cleaning...')
    print(f'Loading data from: {input_filepath}')
    np.seterr(all='raise')

    df = pd.read_parquet(input_filepath)

    df.sort_values(["SK_ID_PREV","DAYS_INSTALMENT"],inplace=True)

    df["INITIAL_VERSION"]= df.groupby("SK_ID_PREV")["NUM_INSTALMENT_VERSION"].transform("first")

    nans_to_drop= (df["AMT_PAYMENT"].isna()) & (df["NUM_INSTALMENT_VERSION"] != df["INITIAL_VERSION"])


    index_to_drop=df[nans_to_drop].index
    df.drop(index=index_to_drop,inplace=True)

    
    df_clean = pd.DataFrame()
    
    # id_prev
    df_clean['id_prev'] = df['SK_ID_PREV']

    # id_curr
    # Passthrough, Identifier, No nulls
    df_clean['id_curr'] = df['SK_ID_CURR']

    # num_instalment_version
    df_clean['num_instalment_version'] = clip_p99_x4_and_fill(df['NUM_INSTALMENT_VERSION'], fill_nulls=False)


    df_clean['initial_version'] = df['INITIAL_VERSION']


    df_clean['num_instalment_number'] = df['NUM_INSTALMENT_NUMBER']

    # days_instalment
    df_clean['days_instalment'] = df['DAYS_INSTALMENT']

    # QA
    # days_entry_payment
    df_clean['days_entry_payment'] = np.where(
        df['DAYS_ENTRY_PAYMENT'] < -2952, 
        df['DAYS_INSTALMENT'] + 30, 
        df['DAYS_ENTRY_PAYMENT']
    )

    # days_entry_payment_is_missing
    df_clean['days_entry_payment_is_missing'] = df['DAYS_ENTRY_PAYMENT'].isnull().astype(int)

    # amt_instalment
    df_clean['amt_instalment'] = clip_p999(df['AMT_INSTALMENT'])

    # amt_payment
    df_clean['amt_payment'] = clip_p999(df['AMT_PAYMENT'])

    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')

def clean_pos_cash_balance(input_filepath: Path, output_filepath: Path):
    print('Starting POS_CASH_balance table cleaning...')
    print(f'Loading data from: {input_filepath}')
    np.seterr(all='raise')

    df = pd.read_parquet(input_filepath)
    
    df = df.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, True])
    df_clean = pd.DataFrame()
    
    # id_prev
    df_clean['id_prev'] = df['SK_ID_PREV']

    # id_curr
    df_clean['id_curr'] = df['SK_ID_CURR']

    # months_balance
    df_clean['months_balance'] = df['MONTHS_BALANCE']

    # count_instalment
    df_clean['count_instalment'] = df['CNT_INSTALMENT']

    # count_instalment_is_missing
    df_clean['count_instalment_is_missing'] = df['CNT_INSTALMENT'].isnull().astype(int)

    # count_instalment_future
    df_clean['count_instalment_future'] = df['CNT_INSTALMENT_FUTURE']

    # name_contract_status
    status_mode = df[df['NAME_CONTRACT_STATUS'] != 'XNA']['NAME_CONTRACT_STATUS'].mode()[0]
    df_clean['name_contract_status'] = np.where(
        df['NAME_CONTRACT_STATUS'] == 'XNA', 
        status_mode, 
        df['NAME_CONTRACT_STATUS']
    )

    start_of_loan_as_active = (df_clean['name_contract_status'] == 'Active') & \
                   (df['CNT_INSTALMENT'].isnull() | df['CNT_INSTALMENT_FUTURE'].isnull())

    pattern_occurrences = start_of_loan_as_active.groupby(df['SK_ID_PREV']).transform('sum')
    
    first_row_of_series = df.groupby('SK_ID_PREV').cumcount() == 0 
    start_of_loan_as_active_mask = start_of_loan_as_active & (pattern_occurrences == 1) & first_row_of_series
    
    df_clean['name_contract_status'] = np.where(
        start_of_loan_as_active_mask, 
        'Signed', 
        df_clean['name_contract_status']
    )

    # sk_dpd
    df_clean['sk_dpd'] = df['SK_DPD']

    # sk_dpd_tecnical
    df_clean['sk_dpd_tecnical'] = (df['SK_DPD'] == 1).astype(int)

    # sk_dpd_severe
    df_clean['sk_dpd_severe'] = (df['SK_DPD'] > 1).astype(int)

    # sk_dpd_def
    df_clean['sk_dpd_def'] = df['SK_DPD_DEF']

    # dpd_def_tecnical
    df_clean['dpd_def_tecnical'] = (df['SK_DPD_DEF'] == 1).astype(int)

    # dpd_def_severe
    df_clean['dpd_def_severe'] = (df['SK_DPD_DEF'] > 1).astype(int)

    # amount_advanced_payment
    df_clean['amount_advanced_payment'] = (df.groupby('SK_ID_PREV')['CNT_INSTALMENT_FUTURE'].shift(1) - df['CNT_INSTALMENT_FUTURE'])
    df_clean['amount_advanced_payment'] = np.where(df_clean['amount_advanced_payment'] > 1, df_clean['amount_advanced_payment'], 0)

    is_completed = df_clean['name_contract_status'] == 'Completed'
    completed_cumsum = is_completed.groupby(df['SK_ID_PREV']).cumsum()

    # flag_is_dead_tail
    df_clean['flag_is_dead_tail'] = (is_completed & (completed_cumsum > 1)).astype(int)

    # flag_delay_tail
    is_before_first_completed = (completed_cumsum == 0)
    
    potential_delay_tail = (
        (df_clean['name_contract_status'] == 'Active') & 
        (df['CNT_INSTALMENT_FUTURE'] == 0) & 
        is_before_first_completed
    )
    
    dpd_in_tail = df['SK_DPD'].where(potential_delay_tail)
    dpd_def_in_tail = df['SK_DPD_DEF'].where(potential_delay_tail)

    dpd_is_constant = dpd_in_tail.groupby(df['SK_ID_PREV']).transform('max') == dpd_in_tail.groupby(df['SK_ID_PREV']).transform('min')
    dpd_def_is_constant = dpd_def_in_tail.groupby(df['SK_ID_PREV']).transform('max') == dpd_def_in_tail.groupby(df['SK_ID_PREV']).transform('min')

    df_clean['flag_delay_tail'] = (potential_delay_tail & dpd_is_constant & dpd_def_is_constant).astype(int)

    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')

def clean_bureau_balance(input_filepath: Path, output_filepath: Path):

    np.seterr(all='raise')
    df = pd.read_parquet(input_filepath)
    df_clean = pd.DataFrame()

    df.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'],inplace=True)
    df_clean["id_bureau"]= df["SK_ID_BUREAU"]
    df_clean["months_balance"]= df["MONTHS_BALANCE"]
    df_clean["status"]= df["STATUS"]


    have_at_least_one_status_closed= df["STATUS"].eq("C").groupby(df["SK_ID_BUREAU"]).transform("any")
    have_recent_balance= df["MONTHS_BALANCE"].gt(-4).groupby(df["SK_ID_BUREAU"]).transform("any")

    is_closed = df["STATUS"] == "C"

    df_clean["closing_month"] = (df["MONTHS_BALANCE"].where(is_closed).groupby(df["SK_ID_BUREAU"]).transform("min"))
    df_clean["non_closed_loan"] = (~have_at_least_one_status_closed)
    df_clean["potential_on_going_loan"] = (~have_at_least_one_status_closed) & (have_recent_balance)
    df_clean["incomplete_sequence"] =  (~have_at_least_one_status_closed) & (~have_recent_balance)

    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')
        
    return df_clean

def clean_previous_application(input_filepath: Path, output_filepath: Path):

    np.seterr(all='raise')
    df = pd.read_parquet(input_filepath)
    df_clean = pd.DataFrame()

    #drop inconsistencies previous application
    df["AMT_CREDIT"].dropna(inplace=True)
    df["PRODUCT_COMBINATION"].dropna(inplace=True)
    index_to_drop=df[(df["AMT_ANNUITY"].isnull()) & (df["CNT_PAYMENT"].notna())].index
    df.drop(index=index_to_drop,inplace=True)

    df_clean["id_prev"] = df["SK_ID_PREV"]
    df_clean["id_curr"] = df["SK_ID_CURR"]
    df_clean["name_contract_type"] = df["NAME_CONTRACT_TYPE"]

    df_clean["amt_annuity"] = clip_p999(df["AMT_ANNUITY"])
    df_clean["amt_annuity_and_cnt_payment_are_missing"]= df["AMT_ANNUITY"].isna().astype(int)

    df_clean["amt_application"] = clip_p999(df["AMT_APPLICATION"])    
    df_clean["amt_credit"] = clip_p999(df["AMT_CREDIT"])  

    df_clean["amt_down_payment"] = clip_p999(df["AMT_DOWN_PAYMENT"],set_negatives_to_nan=True)  
    df_clean["amt_down_payment_is_missing"] = df["AMT_DOWN_PAYMENT"].isna().astype(int)


    df_clean["amt_goods_price"] = clip_p999(df["AMT_GOODS_PRICE"])  
    df_clean["amt_goods_price_is_missing"] = df["AMT_GOODS_PRICE"].isna().astype(int)


    df_clean["week_appr_process_start"] = df["WEEKDAY_APPR_PROCESS_START"]
    df_clean["hour_appr_process_start"] = df["HOUR_APPR_PROCESS_START"]

    df_clean["flag_last_application_for_the_contract"]= df["FLAG_LAST_APPL_PER_CONTRACT"]
    df_clean["flag_last_application_in_day"]= df["NFLAG_LAST_APPL_IN_DAY"]

    df_clean["rate_down_payment"] = clip_p999(df["RATE_DOWN_PAYMENT"],set_negatives_to_nan=True)  
    df_clean["rate_down_payment_is_missing"] = df["RATE_DOWN_PAYMENT"].isna().astype(int)

    df_clean["rate_interesting_is_missing"] = df["RATE_INTEREST_PRIMARY"].isna().astype(int)

    df_clean["rate_privileged_is_missing"] = df["RATE_INTEREST_PRIVILEGED"].isna().astype(int)
    
    df_clean["name_contract_status"] = df["NAME_CONTRACT_STATUS"]

    df_clean["name_cash_loan_purpose"]= agroup_ultra_rare_categories(df["NAME_CASH_LOAN_PURPOSE"],1000)

    df_clean["days_decision"] = df["DAYS_DECISION"]

    df_clean["name_payment_type"] = df["NAME_PAYMENT_TYPE"]

    df_clean["code_reject_reason"] = df["CODE_REJECT_REASON"]

    df_clean["name_type_suite"] = df["NAME_TYPE_SUITE"]
    df_clean["name_type_suite_is_missing"] = df["NAME_TYPE_SUITE"].isna().astype(int)

    df_clean["name_client_type"] = df["NAME_CLIENT_TYPE"]

    df_clean["name_goods_category"] = agroup_ultra_rare_categories(df["NAME_GOODS_CATEGORY"],1000)

    df_clean["name_portfolio"] = df["NAME_PORTFOLIO"]
    df_clean["name_product_type"] = df["NAME_PRODUCT_TYPE"]
    df_clean["channel_type"] = df["CHANNEL_TYPE"]

    df_clean["sellerplace_area"]= clip_p99_x4_and_fill(df["SELLERPLACE_AREA"],fill_nulls=False)
    df_clean["flag_invalid_surface_sellerplace_area"] = (df["SELLERPLACE_AREA"] < 0).astype(int)
    df_clean["flag_have_no_surface_sellerplace_area"] = (df["SELLERPLACE_AREA"] == 0).astype(int)
    df_clean["sellerplace_area_is_missing"] = df["SELLERPLACE_AREA"].isna().astype(int)

    df_clean["name_seller_industry"] = df["NAME_SELLER_INDUSTRY"]

    df_clean["cnt_payment"] = df["CNT_PAYMENT"]

    df_clean["name_yield_group"] = df["NAME_YIELD_GROUP"]

    df_clean["product_combination"] = df["PRODUCT_COMBINATION"]

    df_clean["days_first_drawing_has_sentinel_value"] = (df["DAYS_FIRST_DRAWING"] == 365243).astype(int)
    df_clean["days_first_drawing"] = df["DAYS_FIRST_DRAWING"].mask(df["DAYS_FIRST_DRAWING"] == 365243,np.nan)

    df_clean["days_first_due_has_sentinel_value"] = (df["DAYS_FIRST_DUE"] == 365243).astype(int)
    df_clean["days_first_due"] = df["DAYS_FIRST_DUE"].mask(df["DAYS_FIRST_DUE"] == 365243,np.nan)

    df_clean["days_last_due_1st_version_has_sentinel_value"] = (df["DAYS_LAST_DUE_1ST_VERSION"] == 365243).astype(int)
    df_clean["days_last_due_1st_version"] = df["DAYS_LAST_DUE_1ST_VERSION"].mask(df["DAYS_LAST_DUE_1ST_VERSION"] == 365243,np.nan)

    df_clean["days_last_due_has_sentinel_value"] = (df["DAYS_LAST_DUE"] == 365243).astype(int)
    df_clean["days_last_due"] = df["DAYS_LAST_DUE"].mask(df["DAYS_LAST_DUE"] == 365243,np.nan)

    df_clean["days_termination_has_sentinel_value"] = (df["DAYS_TERMINATION"] == 365243).astype(int)
    df_clean["days_termination"] = df["DAYS_TERMINATION"].mask(df["DAYS_TERMINATION"] == 365243,np.nan)

    df_clean["nflag_insured_on_approval"] = df["NFLAG_INSURED_ON_APPROVAL"]
    df_clean["days_and_insurance_information_are_missing"] = df["DAYS_FIRST_DRAWING"].isna().astype(int)

    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')
        
    return df_clean

def clean_application_train(input_filepath: Path, output_filepath: Path):

    np.seterr(all='raise')
    df = pd.read_parquet(input_filepath)
    df_clean = {}

    #drop inconsistencies
    rows_to_drop = (
        (df["SK_ID_CURR"] == "XNA") | 
        (df["AMT_ANNUITY"].isna()) |
        (df["AMT_GOODS_PRICE"].isna()) |
        (df["NAME_FAMILY_STATUS"] == "Unknown") |  
        (df["CNT_FAM_MEMBERS"].isna()) |
        (df["DAYS_LAST_PHONE_CHANGE"].isna()) 
    )
    

    index_to_drop=df[rows_to_drop].index
    df.drop(index=index_to_drop,inplace=True)

    building_features_names= ["APARTMENTS_AVG", "BASEMENTAREA_AVG","YEARS_BEGINEXPLUATATION_AVG","YEARS_BUILD_AVG","COMMONAREA_AVG","ELEVATORS_AVG","ENTRANCES_AVG"
    ,"FLOORSMAX_AVG","FLOORSMIN_AVG","LANDAREA_AVG","LIVINGAPARTMENTS_AVG","LIVINGAREA_AVG","NONLIVINGAPARTMENTS_AVG","NONLIVINGAREA_AVG","APARTMENTS_MODE","BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE","YEARS_BUILD_MODE","COMMONAREA_MODE","ELEVATORS_MODE","ENTRANCES_MODE","FLOORSMAX_MODE","FLOORSMIN_MODE","LANDAREA_MODE","LIVINGAPARTMENTS_MODE"
    ,"LIVINGAREA_MODE","NONLIVINGAPARTMENTS_MODE","NONLIVINGAREA_MODE","APARTMENTS_MEDI","BASEMENTAREA_MEDI","YEARS_BEGINEXPLUATATION_MEDI","YEARS_BUILD_MEDI","COMMONAREA_MEDI"
    ,"ELEVATORS_MEDI","ENTRANCES_MEDI","FLOORSMAX_MEDI","FLOORSMIN_MEDI","LANDAREA_MEDI","LIVINGAPARTMENTS_MEDI","LIVINGAREA_MEDI","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAREA_MEDI","FONDKAPREMONT_MODE","HOUSETYPE_MODE","TOTALAREA_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE"]
    

    documents_flags_names= ["FLAG_DOCUMENT_"+ str(i) for i in range (2,21)]

    df_clean["id_curr"] = df["SK_ID_CURR"]

    df_clean["target"] = df["TARGET"]

    df_clean["name_contract_type"] = df["NAME_CONTRACT_TYPE"]

    df_clean["code_gender"] = df["CODE_GENDER"]

    df_clean["flag_own_car"] = df["FLAG_OWN_CAR"]

    df_clean["flag_own_realty"] = df["FLAG_OWN_REALTY"]

    capping_value_children=df["CNT_CHILDREN"].quantile(0.99)
    df_clean["cnt_children"] = df["CNT_CHILDREN"].mask(df["CNT_CHILDREN"] > capping_value_children, capping_value_children)

    df_clean["amt_income_total"] = clip_p999(df["AMT_INCOME_TOTAL"])

    df_clean["amt_credit"] = df["AMT_CREDIT"]

    df_clean["amt_annuity"] = df["AMT_ANNUITY"]

    df_clean["amt_goods_price"] = df["AMT_GOODS_PRICE"]
    df_clean["amt_goods_price_is_missing"] = df["AMT_GOODS_PRICE"].isna().astype(int)

    df_clean["name_type_suite"] = df["NAME_TYPE_SUITE"]
    df_clean["name_type_suite_is_missing"] = df["NAME_TYPE_SUITE"].isna().astype(int)


    df_clean["name_income_type"] = df["NAME_INCOME_TYPE"].mask((df["NAME_INCOME_TYPE"] == "Unemployed") | (df["NAME_INCOME_TYPE"]== 'Maternity leave'),"high_risk_inactive").astype("category")

    df_clean["education_type"] = df["NAME_EDUCATION_TYPE"].mask(df["NAME_EDUCATION_TYPE"] == "Academic degree", "Higher education").astype("category")

    df_clean["family_status"] = df["NAME_FAMILY_STATUS"]

    df_clean["housing_type"] = df["NAME_HOUSING_TYPE"]

    df_clean["region_population"] = df["REGION_POPULATION_RELATIVE"]

    df_clean["days_birth"] = df["DAYS_BIRTH"]   

    df_clean["have_sentinel_value_days_employed"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df_clean["days_employed"] = df["DAYS_EMPLOYED"].mask(df["DAYS_EMPLOYED"] == 365243, np.nan)

    df_clean["days_registration"] = df["DAYS_REGISTRATION"]

    df_clean["days_id_publish"] = df["DAYS_ID_PUBLISH"]

    df_clean["own_car_age"] = df["OWN_CAR_AGE"]

    df_clean["own_car_age_is_missing"] = df["OWN_CAR_AGE"].isna().astype(int)

    df_clean["flag_emp_phone"] = df["FLAG_EMP_PHONE"]

    df_clean["flag_cont_mobile"] = df["FLAG_CONT_MOBILE"]

    df_clean["flag_phone"] = df["FLAG_PHONE"]

    df_clean["flag_email"] = df["FLAG_EMAIL"]

    

    mapping_ocupation_type = {"IT staff": "High skill tech staff",
                "HR staff": "Core staff"}
    
    df_clean["occupation_type"] = df["OCCUPATION_TYPE"].replace(mapping_ocupation_type).astype("category")
    df_clean["occupation_type_is_missing"] = df["OCCUPATION_TYPE"].isna().astype(int)

    capping_cnt_fam_members=df["CNT_FAM_MEMBERS"].quantile(0.99)
    df_clean["cnt_family_members"] = df["CNT_FAM_MEMBERS"].mask(df["CNT_FAM_MEMBERS"] > capping_cnt_fam_members, capping_cnt_fam_members)

    df_clean["region_raiting_client"] = df["REGION_RATING_CLIENT"]

    df_clean["region_raiting_client_city"] = df["REGION_RATING_CLIENT_W_CITY"]

    df_clean["weekday_appr_process_start"] = df["WEEKDAY_APPR_PROCESS_START"]

    df_clean["hour_apply_start"] = df["HOUR_APPR_PROCESS_START"]

    df_clean["flag_region_not_live"] = df["REG_REGION_NOT_LIVE_REGION"]
    df_clean["flag_region_not_work"] = df["REG_REGION_NOT_WORK_REGION"]
    df_clean["flag_live_region_not_work"] = df["LIVE_REGION_NOT_WORK_REGION"]
    df_clean["flag_not_live_city"] = df["REG_CITY_NOT_LIVE_CITY"]
    df_clean["flag_city_not_work"] = df["REG_CITY_NOT_WORK_CITY"]
    df_clean["flag_live_city_not_work"] = df["LIVE_CITY_NOT_WORK_CITY"]

    mapping_organization_type = {
        "Religion": "Other",
        "Culture": "Other",
        "Transport: type 1": "Other",
        "Mobile": "Other",
        "Legal Services": "Other",
        "Trade: type 4" : "Other trade",
        "Trade: type 5" : "Other trade",
        "Trade: type 1" : "Other trade",
        "Industry: type 8" : "Other industry",
        "Industry: type 13" : "Other industry",
        "Industry: type 10" : "Other industry",
        "Industry: type 6" : "Other industry",
    }

    df_clean["organization_type"] = df["ORGANIZATION_TYPE"].replace(mapping_organization_type).astype("category")

    df_clean["ext_source_1"] = df["EXT_SOURCE_1"]
    
    df_clean["ext_source_1_is_missing"] = df["EXT_SOURCE_1"].isna().astype(int)

    df_clean["ext_source_2"] = df["EXT_SOURCE_2"]

    df_clean["ext_source_2_is_missing"] = df["EXT_SOURCE_2"].isna().astype(int)
    
    df_clean["ext_source_3"] = df["EXT_SOURCE_3"]

    df_clean["ext_source_3_is_missing"] = df["EXT_SOURCE_3"].isna().astype(int)

    for src in building_features_names:
        dst = src.lower()
        df_clean[dst] = df[src]

    df_clean["obs_30_cnt_social_circle"] = df["OBS_30_CNT_SOCIAL_CIRCLE"]

    df_clean["def_30_cnt_social_circle"] = df["DEF_30_CNT_SOCIAL_CIRCLE"]

    df_clean["obs_60_cnt_social_circle"] = df["OBS_60_CNT_SOCIAL_CIRCLE"]

    df_clean["def_60_cnt_social_circle"] = df["DEF_60_CNT_SOCIAL_CIRCLE"]

    df_clean["info_of_social_circule_is_missing"] = df["OBS_30_CNT_SOCIAL_CIRCLE"].isna().astype(int)

    df_clean["days_last_phone_change"] = df["DAYS_LAST_PHONE_CHANGE"]

    df_clean["flag_document_3"] = df["FLAG_DOCUMENT_3"]
    df_clean["flag_document_5"] = df["FLAG_DOCUMENT_5"]
    df_clean["flag_document_6"] = df["FLAG_DOCUMENT_6"]
    df_clean["flag_document_8"] = df["FLAG_DOCUMENT_8"]
    df_clean["flag_document_9"] = df["FLAG_DOCUMENT_9"]
    df_clean["flag_document_11"] = df["FLAG_DOCUMENT_11"]
    df_clean["flag_document_13"] = df["FLAG_DOCUMENT_13"]
    df_clean["flag_document_14"] = df["FLAG_DOCUMENT_14"]
    df_clean["flag_document_16"] = df["FLAG_DOCUMENT_16"]
    df_clean["flag_document_18"] = df["FLAG_DOCUMENT_18"]

    df_clean["documents_count"] = df[documents_flags_names].sum(axis=1)
    df_clean["amt_req_credit_berau_hour"] = df["AMT_REQ_CREDIT_BUREAU_HOUR"]
    df_clean["amt_req_credit_breau_day"] = df["AMT_REQ_CREDIT_BUREAU_DAY"]
    df_clean["amt_req_credit_breau_week"] = df["AMT_REQ_CREDIT_BUREAU_WEEK"]
    df_clean["amt_req_credit_breau_mon"] = df["AMT_REQ_CREDIT_BUREAU_MON"]
    df_clean["amt_req_credit_breau_qrt"] = df["AMT_REQ_CREDIT_BUREAU_QRT"]
    df_clean["amt_req_credit_breau_year"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"]
    df_clean["client_without_querys"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)


    df_clean_copied = pd.DataFrame(df_clean)
    df_clean_copied.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')
        
    return df_clean_copied

def clean_bureau(input_filepath: Path, output_filepath: Path):
    print('Starting bureau table cleaning...')
    print(f'Loading data from: {input_filepath}')
    np.seterr(all='raise')

    df = pd.read_parquet(input_filepath)
    df_clean = pd.DataFrame()

     #drop inconsistencies
    rows_to_drop = (
        ((df['CREDIT_ACTIVE'] == 'Active') & (df['DAYS_ENDDATE_FACT'].notna())) | 
        ((df['CREDIT_ACTIVE'] == 'Closed') & (df['DAYS_ENDDATE_FACT'].isna())) |
        (df["AMT_CREDIT_SUM"].isna()) 
    )
    index_to_drop=df[rows_to_drop].index
    df.drop(index=index_to_drop,inplace=True)
    
    # id_curr
    df_clean['id_curr'] = df['SK_ID_CURR']

    # bureau_id
    df_clean['id_bureau'] = df['SK_ID_BUREAU']

    # credit_active
    credit_active_status = df['CREDIT_ACTIVE'].replace({'Bad debt': 'Sold'})
    mask_closed = df['DAYS_ENDDATE_FACT'].notnull() & (credit_active_status != 'Active')
    df_clean['credit_active'] = np.where(mask_closed, 'Closed', credit_active_status)

    # credit_currency
    counts = df['CREDIT_CURRENCY'].value_counts()
    currencies_to_keep = counts[counts >= 200].index
    df_clean['credit_currency'] = np.where(df['CREDIT_CURRENCY'].isin(currencies_to_keep), df['CREDIT_CURRENCY'], 'other')

    # days_credit
    df_clean['days_credit'] = df['DAYS_CREDIT']

    # flag_have_credit_day_overdue
    df_clean['flag_have_credit_day_overdue'] = (df['CREDIT_DAY_OVERDUE'] > 0).astype(int)

    # days_credit_enddate
    df_clean['days_credit_enddate'] = df['DAYS_CREDIT_ENDDATE']
    mask_nan_range = (df['DAYS_CREDIT_ENDDATE'] >= 12000) & (df['DAYS_CREDIT_ENDDATE'] <= 31500)
    df_clean.loc[mask_nan_range, 'days_credit_enddate'] = np.nan

    # days_credit_enddate_first_cluster_values
    mask_cluster_1 = (df['DAYS_CREDIT_ENDDATE'] >= 1600) & (df['DAYS_CREDIT_ENDDATE'] < 12000)
    min_x_interval = df.loc[mask_cluster_1, 'DAYS_CREDIT_ENDDATE'].min()
    df_clean['days_credit_enddate_first_cluster_values'] = np.where(
        mask_cluster_1, 
        df['DAYS_CREDIT_ENDDATE'] - min_x_interval, 
        df['DAYS_CREDIT_ENDDATE']
    )

    # days_credit_enddate_is_missing
    df_clean['days_credit_enddate_is_missing'] = df['DAYS_CREDIT_ENDDATE'].notnull().astype(int)

    # days_credit_enddate_closed
    df_clean['days_credit_enddate_closed'] = (df['DAYS_CREDIT_ENDDATE'] < 0).astype(int)

    # days_credit_enddate_firstpositive_cluster
    df_clean['days_credit_enddate_first_positive_cluster'] = (df['DAYS_CREDIT_ENDDATE'] > 1600) & (df['DAYS_CREDIT_ENDDATE'] < 12000)

    # days_credit_enddate_second_positive_cluster
    df_clean['days_credit_enddate_second_positive_cluster'] = (df['DAYS_CREDIT_ENDDATE'] > 12000) & (df['DAYS_CREDIT_ENDDATE'] < 18000)

    # days_credit_enddate_third_positive_cluster
    df_clean['days_credit_enddate_third_positive_cluster'] = (df['DAYS_CREDIT_ENDDATE'] > 27000) & (df['DAYS_CREDIT_ENDDATE'] < 28500)

    # days_credit_enddate_fourth_positive_cluster
    df_clean['days_credit_enddate_fourth_positive_cluster'] = ((df['DAYS_CREDIT_ENDDATE'] > 30000) & (df['DAYS_CREDIT_ENDDATE'] < 31500)).astype(int)

    # days_enddate_fact
    df_clean['days_enddate_fact'] = np.where(df['DAYS_ENDDATE_FACT'] > 3000, 3000, df['DAYS_ENDDATE_FACT'])

    # days_enddate_fact_is_missing
    df_clean['days_enddate_fact_is_missing'] = df['DAYS_ENDDATE_FACT'].isnull().astype(int)

    # amt_credit_max_overdue
    df_clean['amt_credit_max_overdue'] = clip_p999(df['AMT_CREDIT_MAX_OVERDUE'])

    # amt_credit_max_overdue_is_missing
    df_clean['amt_credit_max_overdue_is_missing'] = df['AMT_CREDIT_MAX_OVERDUE'].isnull().astype(int)

    # cnt_credit_prolong
    df_clean['cnt_credit_prolong'] = df['CNT_CREDIT_PROLONG']

    # amt_credit_sum
    df_clean['amt_credit_sum'] = clip_p999(df['AMT_CREDIT_SUM'])

    # amt_credit_sum_debt
    is_negative_mask=df['AMT_CREDIT_SUM_DEBT'] < 0
    df_clean["amt_credit_sum_debt_is_negative"]= np.where(df['AMT_CREDIT_SUM_DEBT'] < 0, df['AMT_CREDIT_SUM_DEBT'] , 0)
    df["AMT_CREDIT_SUM_DEBT"]= df["AMT_CREDIT_SUM_DEBT"].mask(is_negative_mask,0) 
    df_clean['amt_credit_sum_debt'] = clip_p999(df['AMT_CREDIT_SUM_DEBT'])
    

    # amt_credit_sum_debt_is_missing
    df_clean['amt_credit_sum_debt_is_missing'] = df['AMT_CREDIT_SUM_DEBT'].isnull().astype(int)

    # amt_credit_sum_limit
    df_clean['amt_credit_sum_limit'] = df['AMT_CREDIT_SUM_LIMIT']

    # amt_credit_sum_limit_is_missing
    df_clean['amt_credit_sum_limit_is_missing'] = df['AMT_CREDIT_SUM_LIMIT'].isnull().astype(int)

    # amt_credit_sum_limit_is_zero
    df_clean['amt_credit_sum_limit_is_zero'] = (df['AMT_CREDIT_SUM_LIMIT'] == 0).astype(int)

    # amt_credit_sum_limit_short_limit
    df_clean['amt_credit_sum_limit_short_limit'] = ((df['AMT_CREDIT_SUM_LIMIT'] > 0) & (df['AMT_CREDIT_SUM_LIMIT'] < 10000)).astype(int)

    # amt_credit_sum_limit_long_limit
    df_clean['amt_credit_sum_limit_long_limit'] = (df['AMT_CREDIT_SUM_LIMIT'] >= 10000).astype(int)

    # have_amt_credit_sum_overdue
    df_clean['have_amt_credit_sum_overdue'] = (df['AMT_CREDIT_SUM_OVERDUE'] > 0).astype(int)

    # have_amt_credit_sum_overdue_is_missing
    df_clean['amt_credit_sum_overdue_is_missing'] = df['AMT_CREDIT_SUM_OVERDUE'].isnull().astype(int)

    # credit_type
    type_counts = df['CREDIT_TYPE'].value_counts()
    types_to_keep = type_counts[type_counts >= 200].index
    df_clean['credit_type'] = np.where(df['CREDIT_TYPE'].isin(types_to_keep), df['CREDIT_TYPE'], 'other')

    # days_credit_update
    df_clean['days_credit_update'] = np.clip(df['DAYS_CREDIT_UPDATE'], -3000, 0)

    # amt_annuity
    df_clean['amt_annuity'] = df['AMT_ANNUITY']

    # flag_is_missing_amt_annuity
    df_clean['amt_annuity_is_missing'] = df['AMT_ANNUITY'].isnull().astype(int)



    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')