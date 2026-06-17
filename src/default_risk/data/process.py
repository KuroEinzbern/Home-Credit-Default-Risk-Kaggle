
from pathlib import Path

import pandas as pd
import numpy as np


def process_bureau(input_filepath: Path, output_filepath: Path):
    print('Starting bureau table processing...')
    print(f'Loading data from: {input_filepath}')
    bureau_df = pd.read_parquet(input_filepath)

    bureau_df.sort_values(["id_curr", "days_credit"],inplace=True,ascending=False)
    last_three = bureau_df.groupby("id_curr").head(1)
    last_three = last_three.copy()
    last_three["loan_order"] = last_three.groupby("id_curr").cumcount() + 1
    last_three_columns = last_three.pivot(index="id_curr", columns="loan_order")
    last_three_columns.columns =[f"{col}_prev_{rank}" for col, rank in last_three_columns.columns]

    def get_first_mode(x):
        mode_series = x.mode()
        return mode_series.iloc[0] if not mode_series.empty else pd.NA

    bureau_df["log_amt_credit_sum"] = np.log1p(bureau_df["amt_credit_sum"])
    bureau_df["ratio_credit_annuity"]= np.where(bureau_df["amt_annuity"] !=0, bureau_df["amt_credit_sum"] / bureau_df["amt_annuity"] , np.nan )  

    bureau_df["credit_active"]=bureau_df["credit_active"].str.lower()
    bureau_df= pd.get_dummies(bureau_df, columns=["credit_active"], dtype=int)
    

    bureau_aggregated = bureau_df.groupby("id_curr").agg({


        "id_curr": ["count"],
    
        #monetary
        "amt_credit_sum": ["max", "min","sum"],
        "amt_credit_sum_limit": ["max","mean","min"],
        "amt_annuity" : ["max","mean","min"],
        "amt_credit_sum_debt" : ["max","mean","sum"],


        #log_transformed
        "log_amt_credit_sum": ["mean","std"],

        #counters
        "cnt_credit_prolong": ["max","mean","sum"],
        "days_credit_update": ["min","max","mean"], 
        "days_credit": ["min","max","mean"], 
        "days_enddate_fact": ["max"], 

        #other
        "ratio_credit_annuity" : ["max","mean","min"],
        
        #categoricals
        "credit_active_active" : ["mean","sum"],
        "credit_active_closed" : ["mean","sum"],
        "credit_active_sold" : ["mean","sum"],
        "amt_credit_sum_overdue_is_missing": ["mean","sum"],
        "amt_credit_sum_debt_is_negative": ["mean","sum"],
        "days_enddate_fact_is_missing" : ["mean","sum"],
        "flag_have_credit_day_overdue": ["mean","sum"],
        "have_amt_credit_sum_overdue" : ["mean","sum"],
        "amt_credit_sum_limit_is_missing" : ["mean","sum"],
        "amt_credit_sum_limit_is_zero" : ["mean","sum"],
        "amt_annuity_is_missing" : ["mean","sum"],

    })

    bureau_aggregated.columns = [
        f"{col}_{stat.__name__ if callable(stat) else stat}" 
        for col, stat in bureau_aggregated.columns
    ]

    combined_rows = pd.concat([bureau_aggregated, last_three_columns],  axis=1 )
    combined_rows.reset_index(inplace=True)
    combined_rows.to_parquet(output_filepath, index=False)



def process_prev_application(input_filepath: Path, output_filepath: Path) :
    print('Starting previous_application table processing...')
    print(f'Loading data from: {input_filepath}')
    previous_application_df = pd.read_parquet(input_filepath)
    previous_application_df["diff_application_credit"] = previous_application_df["amt_application"] - previous_application_df["amt_credit"]
    previous_application_df["ratio_credit_to_goods"] = previous_application_df["amt_credit"] / (previous_application_df["amt_goods_price"].replace(0,np.nan))
    previous_application_df["total_interest_charged"] = (previous_application_df["amt_annuity"] * previous_application_df["cnt_payment"]) - previous_application_df["amt_credit"]
    
    previous_application_df["ratio_credit_to_annuity"]= previous_application_df["amt_credit"] / (previous_application_df["amt_annuity"].replace(0,np.nan))
    previous_application_to_pivot_df= previous_application_df.drop(columns="id_prev")
    mask_no_final= previous_application_df["flag_last_application_for_the_contract"] == "N"
    previous_application_to_pivot_df= previous_application_to_pivot_df.loc[~mask_no_final]
    previous_application_to_pivot_df.sort_values(["id_curr","days_decision"],inplace=True,ascending=False)
    last_three= previous_application_to_pivot_df.groupby("id_curr").head(3)
    last_three = last_three.copy()
    last_three["loan_order"] = last_three.groupby("id_curr").cumcount() + 1
    df_wide = last_three.pivot(index="id_curr", columns="loan_order")
    df_wide.columns =[f"{col}_prev_{rank}" for col, rank in df_wide.columns]
    df_wide= df_wide.reset_index()
    previous_application_df["name_contract_status"]= previous_application_df["name_contract_status"].str.lower()
    previous_application_df= pd.get_dummies(previous_application_df, columns=["name_contract_status"])
    previous_application_df["log_amt_credit"] = np.log1p(previous_application_df["amt_credit"])
    previous_application_df["log_amt_application"] = np.log1p(previous_application_df["amt_application"])
    previous_application_df["log_amt_annuity"] = np.log1p(previous_application_df["amt_annuity"])
    previous_application_df["log_amt_down_payment"] = np.log1p(previous_application_df["amt_down_payment"])
    previous_application_df["log_amt_goods_price"] = np.log1p(previous_application_df["amt_goods_price"])
    previous_application_df["log_diff_application_credit"] = previous_application_df["log_amt_application"] - previous_application_df["log_amt_credit"]


    agg_metrics_df= previous_application_df.groupby("id_curr").agg({

        #saving the ammount of contract
        "id_prev" : ["count"],
        
        #for log transformated we want to catch the mean and the std (avoiding the impact of the heavy tail from this columns)
        "log_amt_credit": ["mean","std"],   
        "log_amt_application": ["mean","std"],
        "log_amt_down_payment": ["mean","std"],
        "log_amt_goods_price": ["mean","std"],
        "log_amt_annuity": ["mean","std"],

        #for non transformated columns we want to catch the representative values and the acumulated
        "amt_credit": ["max", "min","median","sum"],
        "amt_application": ["max", "min","median","sum"],
        "amt_down_payment": ["max", "min","median","sum"],
        "amt_goods_price": ["max", "min","median","sum"],
        "amt_annuity": ["max", "min","median"],
        "total_interest_charged": ["max", "min","median","mean","std"],

        #others_monetary
        "diff_application_credit": ["max","mean","min","median","sum"],
        "log_diff_application_credit": ["max","mean","min"],
        "rate_down_payment": ["max","mean","std","min","median"],
        "ratio_credit_to_goods" : ["max","mean","std","min","median"],
        "ratio_credit_to_annuity" : ["max","mean","std","min","median"],

        #categoricals
        "name_contract_status_approved": ["mean","sum"],
        "name_contract_status_canceled": ["mean","sum"],
        "name_contract_status_refused": ["mean","sum"],
        "amt_annuity_and_cnt_payment_are_missing" :["mean","sum"],
        "amt_down_payment_is_missing" : ["mean","sum"],
        "nflag_insured_on_approval" : ["mean","sum"],
        "days_and_insurance_information_are_missing": ["mean","sum"],
        "amt_goods_price_is_missing" : ["mean","sum"],
        "rate_down_payment_is_missing" : ["mean","sum"],


        #counters
        "days_decision":["mean","min","max"],
        "cnt_payment":["mean","min","max","sum"]
    })
    agg_metrics_df.columns= [f"{col[0]}_{col[1]}" for col in agg_metrics_df.columns]
    agg_metrics_df= agg_metrics_df.reset_index()
    agg_metrics_df.rename(columns={"id_prev_count": "applications_count"})
    previous_application_ready_to_merge= df_wide.merge(agg_metrics_df,on="id_curr",how="left")
    previous_application_ready_to_merge.to_parquet(output_filepath, index=False)
    return
