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

def clip_p99_x4_and_fill(series, fill_nulls=False):
    s = series.copy()
    p99_x4 = s.quantile(0.99) * 4
    s.loc[s > p99_x4] = p99_x4
    if fill_nulls:
        s = s.fillna(0)
    return s

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

    # amt_drawings_is_present
    df_clean['amt_drawings_is_present'] = df[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].isnull().any(axis=1).astype(int)

    # amt_inst_min_regularity
    df_clean['amt_inst_min_regularity'] = log1p_and_clip_p999(df['AMT_INST_MIN_REGULARITY'])
    
    # amt_inst_min_regularity_is_present
    df_clean['amt_inst_min_regularity_is_present'] = df['AMT_INST_MIN_REGULARITY'].isnull().astype(int)

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
    
    # amt_payment_total_current_is_present
    df_clean['amt_payment_total_current_is_present'] = df['AMT_PAYMENT_TOTAL_CURRENT'].isnull().astype(int)

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

    # days_entry_payment_is_present
    df_clean['days_entry_payment_is_present'] = df['DAYS_ENTRY_PAYMENT'].isnull().astype(int)

    # amt_instalment
    df_clean['amt_instalment'] = log1p_and_clip_p999(df['AMT_INSTALMENT'])

    # amt_payment
    df_clean['amt_payment'] = log1p_and_clip_p999(df['AMT_PAYMENT'])

    df_clean.to_parquet(output_filepath, index=False)
    print(f'Cleaning finished! File saved to: {output_filepath}')
