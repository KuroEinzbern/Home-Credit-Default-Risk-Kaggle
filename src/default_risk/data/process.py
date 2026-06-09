
from pathlib import Path

import pandas as pd
import numpy as np

from default_risk.config import CLEANS_DIR


def process_bureau(input_filepath: Path, output_filepath: Path):
    print('Starting bureau table processing...')
    print(f'Loading data from: {input_filepath}')
    bureau_df = pd.read_parquet(input_filepath)

    bureau_df.sort_values(["id_curr", "days_credit"],inplace=True,ascending=False)
    last_three = bureau_df.groupby("id_curr").head(3)
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
    combined_rows.add_prefix("bureau").to_parquet(output_filepath, index=False)

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

def process_installments_payments(input_filepath: Path, output_filepath: Path):
    print('Starting payment installments table processing...')
    installments_payment_df = pd.read_parquet(input_filepath)
    column_order_reference="days_instalment"
    installments_payment_df.sort_values(["id_prev",column_order_reference,"days_instalment"],inplace=True)

    #we starting catching this because we are probably cutting some parts of the temporal sequence
    installments_payment_df["raw_size_serie"]= installments_payment_df.groupby("id_prev").transform("size")
    installments_payment_df["amount_of_versions_in_sequence"] = installments_payment_df.groupby("id_prev")["num_instalment_version"].transform("nunique")

    #we gonna use the knowlege recolected in the EDA. 
    #For more details look eda_installments_payments.ipynb decisions summary 1#

    #creating auxiliar columns
    installments_payment_df["next_payment_value"] = installments_payment_df.groupby("id_prev")["days_entry_payment"].shift(-1)
    nan_amount= installments_payment_df.groupby("id_prev")["days_entry_payment_is_missing"].transform("sum")

    #defining the mask to separate the differents cases of missings values 
    have_missing_entry_payment_mask= (installments_payment_df["next_payment_value"].isna())
    not_a_deadtail_mask= (installments_payment_df["days_entry_payment"].isna() ) & ( installments_payment_df["next_payment_value"].notna())

    #we want to catch the cases of  dead-tail so we starting filtering that cases with nans but that are not dead tails
    non_dead_tail_nans= installments_payment_df[not_a_deadtail_mask]
    ids_with_nulls_that_are_non_deadtails= non_dead_tail_nans["id_prev"].unique()
    excluding_nans_non_deadtails_mask= ~(installments_payment_df["id_prev"].isin(ids_with_nulls_that_are_non_deadtails))
    series_without_problematic_nans = installments_payment_df[excluding_nans_non_deadtails_mask]

    #now this ID are series where the nans are deadtails, so count nans in "days_entry_payment" o "amt_payment" is calculate
    #the lenght of the deadtail and we save that value in a new column
    ids_with_deadtails= series_without_problematic_nans[series_without_problematic_nans["days_entry_payment"].isna()]["id_prev"].unique()
    installments_payment_df["dead_tail_length"]= np.where(installments_payment_df["id_prev"].isin(ids_with_deadtails),  nan_amount, 0)

    #also in the decision summary of the EDA we define a criteria to incomplete series 
    starting_instalment_number= installments_payment_df.groupby("id_prev")["num_instalment_number"].transform("first")
    starting_date= installments_payment_df.groupby("id_prev")["days_instalment"].transform("first")

    installments_payment_df["is_potentially_incomplete_sequence"] = ((starting_instalment_number >  1)  & (starting_date < -2890 ))

    potentially_on_going_id = installments_payment_df[installments_payment_df["days_instalment"] > (- 33)]["id_prev"].unique()
    installments_payment_df["potentially_on_going"]= installments_payment_df["id_prev"].isin(potentially_on_going_id)   

    installments_payment_df["diff_expected_received"]= installments_payment_df["amt_instalment"] -  installments_payment_df["amt_payment"]
    installments_payment_df["diff_deadline_factical_payment"]= installments_payment_df["days_instalment"] -  installments_payment_df["days_entry_payment"]
    installments_payment_df["days_of_delinquency"]= installments_payment_df["diff_deadline_factical_payment"].clip(lower=0)
    installments_payment_df["is_delinquency"] = installments_payment_df["days_of_delinquency"] > 0 

    next_installment_number = installments_payment_df.groupby("id_prev")["num_instalment_number"].shift(-1)
    next_version_number = installments_payment_df.groupby("id_prev")["num_instalment_version"].shift(-1)
    repeated_installment_mask= (installments_payment_df ["num_instalment_number"] == next_installment_number)
    underpayment_mask= (installments_payment_df[ "amt_payment" ] < installments_payment_df ["amt_instalment"]) & (installments_payment_df[ "amt_payment" ] == 0)
    full_payment_mask= installments_payment_df[ "amt_payment" ] == installments_payment_df ["amt_instalment"] 
    installments_payment_df[ "repeated_for_underpayment" ] = (repeated_installment_mask) & (underpayment_mask)
    installments_payment_df[ "repeated_for_reschedule" ] = (repeated_installment_mask) & (full_payment_mask)
    installments_payment_df["repeated_for_payment_in_advance"] = (repeated_installment_mask) & (installments_payment_df[ "amt_payment" ] == 0) & (installments_payment_df["days_of_delinquency"] == 0)
    installments_payment_df["log_amt_instalment"]= np.log1p(installments_payment_df ["amt_instalment"] )
    installments_payment_df["log_amt_payment"]= np.log1p(installments_payment_df ["amt_payment"] )

    agg_metrics_df= installments_payment_df.groupby("id_prev").agg({

        #static values calculated for the entire squenece
        "raw_size_serie" : ["first"],
        "dead_tail_length" : ["first"],
        "amount_of_versions_in_sequence" : ["first"],
        "is_potentially_incomplete_sequence" : ["first"],
        "potentially_on_going" : ["first"],

        #for log transformated we want to catch the mean and the std (avoiding the impact of the heavy tail from this columns)
        "log_amt_instalment": ["mean","std"],   
        "log_amt_payment": ["mean","std"],

        #natural scale
        "amt_instalment": ["max", "min","median","sum"],
        "amt_payment": ["max", "min","median","sum"],

        #computed_differences
        "diff_expected_received": ["max", "min","median","sum"],
        "diff_deadline_factical_payment": ["max", "min","median","sum"],


        #categoricals
        "repeated_for_underpayment": ["mean","sum"],
        "repeated_for_reschedule": ["mean","sum"],
        "repeated_for_payment_in_advance": ["mean","sum"],
        "is_delinquency" : ["mean","sum"],


        #counters
        "days_of_delinquency":["mean","max","sum"],
    })

    agg_metrics_df.columns = [
        f"instalments_{col[0]}" if col[1] == "first" else f"instalments_{col[0]}_{col[1]}"
        for col in agg_metrics_df.columns
    ]

    agg_metrics_df = agg_metrics_df.reset_index()

    agg_metrics_df["instalments_completion_ratio"] = np.where( agg_metrics_df["instalments_amt_instalment_sum"] > 0, agg_metrics_df["instalments_amt_payment_sum"] / agg_metrics_df["instalments_amt_instalment_sum"], 1.0 )  #if the debt is 0 or negative we assume competitud (1)                                                               )
    agg_metrics_df.to_parquet(output_filepath)

def process_previous_application(input_filepath: Path, output_filepath: Path):
    previous_application_df = pd.read_parquet(input_filepath)

    #creating columns before aggregation
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


    #we will calculate aggregations per client for differents tables so we will separate dictionaries per table
    agg_from_prev_app_dict= {

        #saving the ammount of contract
        "id_prev" : ["count"],
        
        #for log transformated we want to catch the mean and the std (avoiding the impact of the heavy tail from this columns)
        "log_amt_credit": ["mean","std"],   
        "log_amt_application": ["mean","std"],
        "log_amt_down_payment": ["mean","std"],
        "log_amt_goods_price": ["mean","std"],
        "log_amt_annuity": ["mean","std"],
        "log_total_interest_charged" : ["mean","std"],

        #for non transformated columns we want to catch the representative values and the acumulated
        "amt_credit": ["max", "min","median","sum"],
        "amt_application": ["max", "min","median","sum"],
        "amt_down_payment": ["max", "min","median","sum"],
        "amt_goods_price": ["max", "min","median","sum"],
        "amt_annuity": ["max", "min","median"],
        "total_interest_charged": ["max", "min","median"],

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
    }

    agg_from_instalment_payment_dict= {
        "instalments_potentially_on_going" : ["sum"],
        "instalments_is_potentially_incomplete_sequence" : ["mean","sum"],
        "instalments_dead_tail_length" : ["mean","max"],
        "instalments_amt_instalment_sum" : ["mean","sum","max"],
        "instalments_amt_payment_sum" : ["mean", "sum", "max"],
        "instalments_days_of_delinquency_max": ["max"],
        "instalments_days_of_delinquency_mean": ["mean", "max"],
        "instalments_is_delinquency_sum" : ["sum"] ,
        "instalments_is_delinquency_mean" : ["mean"],
        "instalments_repeated_for_underpayment_sum" : ["sum"],
        "instalments_repeated_for_underpayment_mean" : ["mean"],
        "instalments_repeated_for_reschedule_sum" : ["sum"],
        "instalments_repeated_for_reschedule_mean" : ["mean"],
        "instalments_diff_expected_received_sum" : ["sum"]
    }

    final_dict_for_agg= agg_from_prev_app_dict | agg_from_instalment_payment_dict

    previous_application_df["log_amt_credit"] = np.log1p(previous_application_df["amt_credit"])
    previous_application_df["log_amt_application"] = np.log1p(previous_application_df["amt_application"])
    previous_application_df["log_amt_annuity"] = np.log1p(previous_application_df["amt_annuity"])
    previous_application_df["log_amt_down_payment"] = np.log1p(previous_application_df["amt_down_payment"])
    previous_application_df["log_amt_goods_price"] = np.log1p(previous_application_df["amt_goods_price"])
    previous_application_df["log_diff_application_credit"] = previous_application_df["log_amt_application"] - previous_application_df["log_amt_credit"]
    previous_application_df["log_total_interest_charged"] = np.log1p(previous_application_df["total_interest_charged"])



    agg_metrics_df= previous_application_df.groupby("id_curr").agg(final_dict_for_agg)
    agg_metrics_df.columns = [
        f"{col[0]}_{col[1]}" if str(col[0]).startswith("instalments_") else f"previous_application_{col[0]}_{col[1]}" 
        for col in agg_metrics_df.columns
    ]
    agg_metrics_df= agg_metrics_df.reset_index()

    agg_metrics_df.rename(columns={"id_prev_count": "applications_count"},inplace=True)
    previous_application_ready_to_merge= df_wide.merge(agg_metrics_df,on="id_curr",how="left")

    cols_to_fix = [
        "instalments_potentially_on_going_sum",
        "instalments_is_potentially_incomplete_sequence_sum",
        "instalments_is_potentially_incomplete_sequence_mean",
        "instalments_is_delincuency_sum",
        "instalments_is_delincuency_mean"
    ]

    for col in cols_to_fix:
        if col in previous_application_ready_to_merge.columns:
            previous_application_ready_to_merge[col] = pd.to_numeric(previous_application_ready_to_merge[col], errors='coerce')

    previous_application_ready_to_merge.to_parquet(output_filepath, index=False)
    pass
