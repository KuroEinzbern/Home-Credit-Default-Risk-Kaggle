import pandas as pd
from scipy.stats import trim_mean       
from IPython.display import display
import numpy as np
import default_risk.config as cfg
import json

def recreate_and_sort_the_serie_given_ids(ids : pd.Series, df : pd.DataFrame,column_time_name) -> pd.DataFrame :
    recreated_series= df[df["SK_ID_PREV"].isin(ids)]
    return recreated_series.sort_values(["SK_ID_PREV",column_time_name])
     

def recreate_and_sort_series_given_rows(rows : pd.DataFrame, df : pd.DataFrame, column_time_name : str) -> pd.DataFrame :
    id_rows= rows["SK_ID_PREV"].unique()
    return recreate_and_sort_the_serie_given_ids(id_rows, df,column_time_name)


def check_invariant(invariant_mask,text_logic_invariant_mask,data_frame_size):
    negative_balance= (invariant_mask).sum()
    print(str(negative_balance) + " of cases where " + text_logic_invariant_mask)
    positive_balance_invariant_porcentaje= ((invariant_mask).sum() * 100) / data_frame_size
    print("that represent a " + str(positive_balance_invariant_porcentaje) + "%" + " of cases with violation of this invariant \n")