import pandas as pd
import numpy as np
from pathlib import Path

def log1p_and_clip_p999(series, set_negatives_to_nan=False):
    s = series.copy()
    if set_negatives_to_nan:
        s.loc[s < 0] = np.nan
    p99_9 = s.quantile(0.999)
    s.loc[s > p99_9] = p99_9
    return np.log1p(s + 1)

def clip_p99_x4_and_fill(series, fill_nulls=False):
    s = series.copy()
    p99_x4 = s.quantile(0.99) * 4
    s.loc[s > p99_x4] = p99_x4
    if fill_nulls:
        s = s.fillna(0)
    return s

def clean_credit_card_balance(input_filepath: Path, output_filepath: Path):
    print('Starting credit_card_balance table cleaning...')

    raw_data_path = input_filepath / 'credit_card_balance.csv'
    print(f'Loading data from: {raw_data_path}')
    df = pd.read_csv(raw_data_path)
    df_clean = pd.DataFrame()

    # SK_ID_PREV
    df_clean['id_prev'] = df['SK_ID_PREV']
    
    # SK_ID_CURR
    df_clean['id_curr'] = df['SK_ID_CURR']
    
    # MONTHS_BALANCE
    df_clean['months_balance'] = df['MONTHS_BALANCE']

    # AMT_BALANCE
    df_clean['amt_balance'] = np.log1p(df['AMT_BALANCE'] + 1)
    
    # AMT_CREDIT_LIMIT_ACTUAL
    df_clean['amt_credit_limit_actual'] = np.log1p(df['AMT_CREDIT_LIMIT_ACTUAL'] + 1)

    # AMT_DRAWINGS_CURRENT
    amt_drawings_curr = df['AMT_DRAWINGS_CURRENT'].copy()
    amt_drawings_curr.loc[amt_drawings_curr < 0] = amt_drawings_curr.median()
    df_clean['amt_drawings_current'] = np.log1p(amt_drawings_curr + 1)

    # AMT_DRAWINGS_ATM_CURRENT
    df_clean['amt_drawings_amt_current'] = log1p_and_clip_p999(df['AMT_DRAWINGS_ATM_CURRENT'], set_negatives_to_nan=True)
    
    # AMT_DRAWINGS_OTHER_CURRENT
    df_clean['amt_drawings_other_current'] = log1p_and_clip_p999(df['AMT_DRAWINGS_OTHER_CURRENT'])
    
    # AMT_DRAWINGS_POS_CURRENT
    df_clean['amt_drawings_pos_current'] = log1p_and_clip_p999(df['AMT_DRAWINGS_POS_CURRENT'])

    # AMT_DRAWINGS_ATM_CURRENT, AMT_DRAWINGS_OTHER_CURRENT, AMT_DRAWINGS_POS_CURRENT
    df_clean['amt_drawings_is_present'] = df[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].isnull().any(axis=1).astype(int)

    # AMT_INST_MIN_REGULARITY
    df_clean['amt_inst_min_regularity'] = log1p_and_clip_p999(df['AMT_INST_MIN_REGULARITY'])
    df_clean['amt_inst_min_regularity_is_present'] = df['AMT_INST_MIN_REGULARITY'].isnull().astype(int)

    # AMT_PAYMENT_CURRENT
    df_clean['amt_payment_current'] = log1p_and_clip_p999(df['AMT_PAYMENT_CURRENT'])
    df_clean['amt_payment_current_is_present'] = df['AMT_PAYMENT_CURRENT'].isnull().astype(int)

    # AMT_PAYMENT_TOTAL_CURRENT
    df_clean['amt_payment_total_current'] = log1p_and_clip_p999(df['AMT_PAYMENT_TOTAL_CURRENT'])
    df_clean['amt_payment_total_current_is_present'] = df['AMT_PAYMENT_TOTAL_CURRENT'].isnull().astype(int)

    # AMT_RECEIVABLE_PRINCIPAL
    s_principal = df['AMT_RECEIVABLE_PRINCIPAL'].copy()
    df_clean['amt_reciavable_principal_positive_balance'] = np.where(s_principal < 0, s_principal * -1, 0)
    s_principal_clipped = np.where(s_principal < 0, 0, s_principal)
    df_clean['amt_recivable_principal'] = np.log1p(s_principal_clipped + 1)

    # AMT_RECIVABLE
    s_recivable = df['AMT_RECIVABLE'].copy()
    df_clean['amt_recivable_positive_balance'] = np.where(s_recivable < 0, s_recivable * -1, 0)
    s_recivable_clipped = np.where(s_recivable < 0, 0, s_recivable)
    df_clean['amt_recivable'] = np.log1p(s_recivable_clipped + 1)

    # AMT_TOTAL_RECEIVABLE
    s_total = df['AMT_TOTAL_RECEIVABLE'].copy()
    df_clean['amt_total_recivable_positive_balance'] = np.where(s_total < 0, s_total * -1, 0)
    s_total_clipped = np.where(s_total < 0, 0, s_total)
    df_clean['amt_total_recivable'] = np.log1p(s_total_clipped + 1)

    # CNT_DRAWINGS_CURRENT
    df_clean['cnt_drawings_current'] = clip_p99_x4_and_fill(df['CNT_DRAWINGS_CURRENT'], fill_nulls=False)
    
    # CNT_DRAWINGS_ATM_CURRENT
    df_clean['cnt_drawings_amt_current'] = clip_p99_x4_and_fill(
        df['CNT_DRAWINGS_ATM_CURRENT'], 
        fill_nulls=True
    )

    # CNT_DRAWINGS_OTHER_CURRENT
    df_clean['cnt_drawings_other_current'] = clip_p99_x4_and_fill(
        df['CNT_DRAWINGS_OTHER_CURRENT'], 
        fill_nulls=True
    )

    # CNT_DRAWINGS_POS_CURRENT
    df_clean['cnt_drawings_pos_current'] = clip_p99_x4_and_fill(
        df['CNT_DRAWINGS_POS_CURRENT'], 
        fill_nulls=True
    )
        
    # CNT_DRAWINGS_ATM_CURRENT, CNT_DRAWINGS_OTHER_CURRENT, CNT_DRAWINGS_POS_CURRENT
    df_clean['cnt_drawings_are_present'] = df[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT']].isnull().any(axis=1).astype(int)

    # CNT_INSTALMENT_MATURE_CUM
    df_clean['cnt_instalment_mature_cum_is_originally_present'] = df['CNT_INSTALMENT_MATURE_CUM'].isnull().astype(int)
    
    df_temp = df[['SK_ID_PREV', 'MONTHS_BALANCE', 'CNT_INSTALMENT_MATURE_CUM']].copy()
    df_temp = df_temp.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    
    df_temp['CNT_INSTALMENT_MATURE_CUM'] = (
        df_temp.groupby('SK_ID_PREV')['CNT_INSTALMENT_MATURE_CUM']
        .apply(lambda group: group.interpolate(method='linear').fillna(0))
    ).reset_index(level=0, drop=True)
    
    df_temp = df_temp.sort_index()
    df_clean['cnt_instalment_mature_cum'] = df_temp['CNT_INSTALMENT_MATURE_CUM']

    # NAME_CONTRACT_STATUS
    contract_status = df['NAME_CONTRACT_STATUS'].copy()
    counts = contract_status.value_counts()
    minority_cats = counts[counts < 1000].index
    majority_cat = counts.index[0] 
    contract_status = contract_status.replace(minority_cats, majority_cat)
    df_clean['name_contract_status'] = contract_status

    # SK_DPD
    dpd_col_sk_dpd = df['SK_DPD']
    df_clean['sk_dpd'] = np.log1p(dpd_col_sk_dpd + 1)
    df_clean['sk_dpd_tecnical'] = (dpd_col_sk_dpd == 1).astype(int)
    df_clean['sk_dpd_severe'] = (dpd_col_sk_dpd > 1).astype(int)

    # SK_DPD_DEF
    dpd_col_sk_dpd_def = df['SK_DPD_DEF']
    df_clean['sk_dpd_def'] = np.log1p(dpd_col_sk_dpd_def + 1)
    df_clean['sk_dpd_def_tecnical'] = (dpd_col_sk_dpd_def == 1).astype(int)
    df_clean['sk_dpd_def_severe'] = (dpd_col_sk_dpd_def > 1).astype(int)
    
    # 2. Saving data
    processed_data_path = output_filepath / '1.0-credit_card_balance.csv'
    df_clean.to_csv(processed_data_path, index=False)
    print(f'Cleaning finished! File saved to: {processed_data_path}')

def main():
    clean_credit_card_balance(Path('../../data/raw'), Path('../../data/interim'))

if __name__ == "__main__":
    main()