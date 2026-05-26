import pandas as pd
import numpy as np
from pathlib import Path

def log1p_and_clip_p999(series, set_negatives_to_nan=False):
    s = series.copy()
    if set_negatives_to_nan:
        s.loc[s < 0] = np.nan
    p99_9 = s.quantile(0.999)
    s.loc[s > p99_9] = p99_9
    return np.log1p(s)

def clip_p999(series):
    s = series.copy()
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
    df_clean['amt_positive_balance'] = np.log1p(df['AMT_BALANCE'].clip(lower=0))

    # amt_negative_balance
    df_clean['amt_negative_balance'] = np.log1p(df['AMT_BALANCE'].clip(upper=0).abs())
    
    # diff_total_receivable_balance
    diff_total_balance = df['AMT_TOTAL_RECEIVABLE'] - df['AMT_BALANCE']
    df_clean['diff_total_receivable_balance'] = np.sign(diff_total_balance) * np.log1p(np.abs(diff_total_balance))

    # amt_credit_limit_actual
    df_clean['amt_credit_limit_actual'] = np.log1p(df['AMT_CREDIT_LIMIT_ACTUAL'])

    # amt_drawings_current
    amt_drawings_curr = df['AMT_DRAWINGS_CURRENT'].copy()
    amt_drawings_curr.loc[amt_drawings_curr < 0] = amt_drawings_curr.median()
    df_clean['amt_drawings_current'] = np.log1p(amt_drawings_curr)

    # amt_drawings_amt_current
    df['AMT_DRAWINGS_ATM_CURRENT'] = df['AMT_DRAWINGS_ATM_CURRENT'].fillna(0)
    df_clean['amt_drawings_amt_current'] = log1p_and_clip_p999(df['AMT_DRAWINGS_ATM_CURRENT'], set_negatives_to_nan=True)
    
    # amt_drawings_other_current
    df['AMT_DRAWINGS_OTHER_CURRENT'] = df['AMT_DRAWINGS_OTHER_CURRENT'].fillna(0)
    df_clean['amt_drawings_other_current'] = log1p_and_clip_p999(df['AMT_DRAWINGS_OTHER_CURRENT'])
    
    # amt_drawings_pos_current
    df['AMT_DRAWINGS_POS_CURRENT'] = df['AMT_DRAWINGS_POS_CURRENT'].fillna(0)
    df_clean['amt_drawings_pos_current'] = log1p_and_clip_p999(df['AMT_DRAWINGS_POS_CURRENT'])

    # amt_drawings_is_missing
    df_clean['amt_drawings_is_missing'] = df[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].isnull().any(axis=1).astype(int)

    # amt_inst_min_regularity
    df_clean['amt_inst_min_regularity'] = log1p_and_clip_p999(df['AMT_INST_MIN_REGULARITY'])
    
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
    df_clean['inconsistency_gap'] = np.where(gap_condition, np.log1p(gap_values.clip(lower=0)), 0)

    # diff_payment_current_total
    df_clean['diff_payment_current_total'] = np.log1p(df['AMT_PAYMENT_CURRENT'] - df['AMT_PAYMENT_TOTAL_CURRENT']+1)

    # amt_payment_total_current
    df_clean['amt_payment_total_current'] = log1p_and_clip_p999(df['AMT_PAYMENT_TOTAL_CURRENT'])
    
    # amt_payment_total_current_is_missing
    df_clean['amt_payment_total_current_is_missing'] = df['AMT_PAYMENT_TOTAL_CURRENT'].isnull().astype(int)

    # amt_recivable_principal
    s_principal_clipped = np.where(df['AMT_RECEIVABLE_PRINCIPAL'] < 0, 0, df['AMT_RECEIVABLE_PRINCIPAL'])
    df_clean['amt_recivable_principal'] = np.log1p(s_principal_clipped + 1)

    # amt_reciavable_principal_positive_balance
    df_clean['amt_reciavable_principal_positive_balance'] = np.where(df['AMT_RECEIVABLE_PRINCIPAL'] < 0, df['AMT_RECEIVABLE_PRINCIPAL'] * -1, 0)

    # amt_recivable
    s_recivable_clipped = np.where(df['AMT_RECIVABLE'] < 0, 0, df['AMT_RECIVABLE'])
    df_clean['amt_recivable'] = np.log1p(s_recivable_clipped)

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
    df_clean['sk_dpd'] = np.log1p(df['SK_DPD'])
    
    # sk_dpd_tecnical
    df_clean['sk_dpd_tecnical'] = (df['SK_DPD'] == 1).astype(int)
    
    # sk_dpd_severe
    df_clean['sk_dpd_severe'] = (df['SK_DPD'] > 1).astype(int)

    # sk_dpd_def
    df_clean['sk_dpd_def'] = np.log1p(df['SK_DPD_DEF'])
    
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
    
    df_clean = pd.DataFrame()
    
    # sk_id_prev
    df_clean['sk_id_prev'] = df['SK_ID_PREV']

    # sk_id_curr
    # Passthrough, Identifier, No nulls
    df_clean['sk_id_curr'] = df['SK_ID_CURR']

    # num_instalment_version
    df_clean['num_instalment_version'] = clip_p99_x4_and_fill(df['NUM_INSTALMENT_VERSION'], fill_nulls=False)

    # num_instalment_number
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
    df_clean['amt_instalment'] = log1p_and_clip_p999(df['AMT_INSTALMENT'])

    # amt_payment
    df_clean['amt_payment'] = log1p_and_clip_p999(df['AMT_PAYMENT'])

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
    df_clean['sk_dpd'] = np.log1p(df['SK_DPD'])

    # sk_dpd_tecnical
    df_clean['sk_dpd_tecnical'] = (df['SK_DPD'] == 1).astype(int)

    # sk_dpd_severe
    df_clean['sk_dpd_severe'] = (df['SK_DPD'] > 1).astype(int)

    # sk_dpd_def
    df_clean['sk_dpd_def'] = np.log1p(df['SK_DPD_DEF'])

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

    df_clean["amt_annuity"] = log1p_and_clip_p999(df["AMT_ANNUITY"])
    df_clean["amt_application"] = log1p_and_clip_p999(df["AMT_APPLICATION"])    
    df_clean["amt_credit"] = log1p_and_clip_p999(df["AMT_CREDIT"])  



    df_clean["amt_down_payment"] = log1p_and_clip_p999(df["AMT_DOWN_PAYMENT"],set_negatives_to_nan=True)  
    df_clean["amt_down_payment_is_missing"] = df["AMT_DOWN_PAYMENT"].isna().astype(int)


    df_clean["amt_goods_price"] = log1p_and_clip_p999(df["AMT_GOODS_PRICE"])  
    df_clean["amt_goods_price_is_missing"] = df["AMT_GOODS_PRICE"].isna().astype(int)


    df_clean["week_appr_process_start"] = df["WEEKDAY_APPR_PROCESS_START"]
    df_clean["hour_appr_process_start"] = df["HOUR_APPR_PROCESS_START"]

    df_clean["flag_last_application_for_the_contract"]= df["FLAG_LAST_APPL_PER_CONTRACT"]
    df_clean["flag_last_application_in_day"]= df["NFLAG_LAST_APPL_IN_DAY"]

    df_clean["rate_down_payment"] = log1p_and_clip_p999(df["RATE_DOWN_PAYMENT"],set_negatives_to_nan=True)  
    df_clean["down_payment_is_missing"] = df["RATE_DOWN_PAYMENT"].isna().astype(int)

    df_clean["rate_interesting_is_missing"] = df["RATE_INTEREST_PRIMARY"].isna().astype(int)

    df_clean["rate_privileged_is_missing"] = df["RATE_INTEREST_PRIVILEGED"].isna().astype(int)


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

    df_clean["days_ 0rmination_has_sentinel_value"] = (df["DAYS_TERMINATION"] == 365243).astype(int)
    df_clean["days_termination"] = df["DAYS_TERMINATION"].mask(df["DAYS_TERMINATION"] == 365243,np.nan)

    df_clean["nflag_insured_on_approval"] = df["NFLAG_INSURED_ON_APPROVAL"]
    df_clean["days_and_insurance_information_are_missing"] = df["DAYS_FIRST_DRAWING"].isna().astype(int)

    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')
        
    return df_clean


def clean_application_train(input_filepath: Path, output_filepath: Path):

    np.seterr(all='raise')
    df = pd.read_parquet(input_filepath)
    df_clean = pd.DataFrame()

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



    df_clean["sk_id_curr"] = df["SK_ID_CURR"]

    df_clean["target"] = df["TARGET"]

    df_clean["name_contract_type"] = df["NAME_CONTRACT_TYPE"]

    df_clean["flag_own_car"] = df["FLAG_OWN_CAR"]

    df_clean["flag_own_realty"] = df["FLAG_OWN_REALTY"]

    capping_value_children=df["CNT_CHILDREN"].quantile(0.99)
    df_clean["cnt_children"] = df["CNT_CHILDREN"].mask(df["CNT_CHILDREN"] > capping_value_children, capping_value_children)

    df_clean["amt_income_total"] = clip_p999(df["AMT_INCOME_TOTAL"])
    
    df_clean["amt_goods_price"] = df["AMT_GOODS_PRICE"]


    df_clean["amt_goods_price"] = df["NAME_TYPE_SUITE"]
    df_clean["amt_goods_price_is_missing"] = df["NAME_TYPE_SUITE"].isna().astype(int)



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

    df_clean["organization_type"] = df["ORGANIZATION_TYPE"].replace(mapping_ocupation_type).astype("category")

    df_clean["ext_source_1"] = df["EXT_SOURCE_1"]
    
    df_clean["ext_source_1_is_missing"] = df["EXT_SOURCE_1"].isna().astype(int)

    df_clean["ext_source_2"] = df["EXT_SOURCE_2"]

    df_clean["ext_source_2_is_missing"] = df["EXT_SOURCE_1"].isna().astype(int)
    
    df_clean["ext_source_3"] = df["EXT_SOURCE_3"]

    df_clean["ext_source_3_is_missing"] = df["EXT_SOURCE_1"].isna().astype(int)

    df_clean["obs_30_cnt_social_circle"] = df["OBS_30_CNT_SOCIAL_CIRCLE"]

    df_clean["def_30_cnt_social_circle"] = df["DEF_30_CNT_SOCIAL_CIRCLE"]

    df_clean["obs_60_cnt_social_circle"] = df["OBS_60_CNT_SOCIAL_CIRCLE"]

    df_clean["def_60_cnt_social_circle"] = df["DEF_60_CNT_SOCIAL_CIRCLE"]

    df_clean["info_of_social_circule_is_missing"] = df["OBS_30_CNT_SOCIAL_CIRCLE"].isna().astype(int)

    df_clean["days_last_phone_change"] = df["DAYS_LAST_PHONE_CHANGE"].isna().astype(int)

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

    df_clean["amt_req_credit_berau_hour"] = df["AMT_REQ_CREDIT_BUREAU_HOUR"]
    df_clean["amt_req_credit_breau_day"] = df["AMT_REQ_CREDIT_BUREAU_DAY"]
    df_clean["amt_req_credit_breau_week"] = df["AMT_REQ_CREDIT_BUREAU_WEEK"]
    df_clean["amt_req_credit_breau_mon"] = df["AMT_REQ_CREDIT_BUREAU_MON"]
    df_clean["amt_req_credit_breau_qrt"] = df["AMT_REQ_CREDIT_BUREAU_QRT"]
    df_clean["amt_req_credit_breau_year"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"]
    df_clean["client_without_querys"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)


    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')
        
    return df_clean