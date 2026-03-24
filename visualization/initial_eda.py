import pandas as pd
from scipy.stats import trim_mean       
from IPython.display import display
import numpy as np
import json


def eda_per_table_persisting_result_html(df: pd.DataFrame, schema,table_name, target_aware: bool):
    results= eda_per_table(df,schema,table_name,target_aware)
    with open("eda_report.html", "w", encoding="utf-8") as f:
            f.write("<html><body>")
            for key,value in results.items() :
                  f.write(f"<h2>{key}</h2>")
                  f.write("<pre>")
                  for inside_keys,data_frames in value.items():
                        f.write(f"<h3>{inside_keys}</h3>")
                        f.write("</pre>")
                        f.write(data_frames.to_html(index=False))
                        f.write("</pre>")
            f.write("</body></html>")



def eda_per_table_printing_results(df: pd.DataFrame,schema: dict, table_name: str, target_aware: bool):
    results=eda_per_table(df,schema,table_name,target_aware)
    for key,values in results.items() :
        print("--------------------------------------")
        print(key)
        print_dataframes(values)


def print_dataframes(dicts):
    for key,a_dataframe in dicts.items():
        print(key)
        display(a_dataframe)
    return


def eda_per_table(df: pd.DataFrame,schema: dict,table_name, target_aware: bool) -> dict :
    results={}
    for col in df.columns:
        dict_of_dataframes,column_name= eda_per_column(df,schema,table_name,col,target_aware)
        results[column_name]=dict_of_dataframes
    return results


def eda_per_column(df: pd.DataFrame,schema: dict,table_name,column_name, target_aware: bool):
    dict_of_dataframes={}
    if(is_categorical(schema,table_name,column_name)):
        dict_of_dataframes= basic_eda_per_column_categorical(df,column_name, target_aware)
    else:
        dict_of_dataframes=basic_eda_per_column_numerical(df,column_name, target_aware) 
    return dict_of_dataframes,column_name


def is_categorical(schema: dict,table_name,column_name):
    try:
            col_type = schema[table_name][column_name]["type"]
            return col_type == "categorical"
    except KeyError:
        print(f"Warning: '{column_name}' not found in schema for '{table_name}'. Defaulting to categorical.")
        return True


def basic_eda_per_column_numerical(df: pd.DataFrame, column_name, target_aware: bool) -> dict: 
    column=df[column_name]
    dict_to_return={}
    basic_data_dict={}

    mean=column.mean()
    median=column.median()
    standar_deviation=column.std()

    basic_data_dict["min"]=column.min()
    basic_data_dict["max"]=column.max()
    basic_data_dict["mean"]=mean

    basic_data_dict["trim_mean"]= trim_mean(column.dropna(),proportiontocut=0.1)
    basic_data_dict["median"]= median
    basic_data_dict["standard_deviation"]=standar_deviation
    basic_data_dict["standard_error"]=column.sem()
    if(mean != 0):
        basic_data_dict["coefficient_of_variation"]= standar_deviation / abs(mean)

    basic_data_dataframe=pd.DataFrame([basic_data_dict])
    distribution_metrics_dataframe=pd.DataFrame([get_distribution_metrics(df,column_name)])

    dict_to_return["basic_data"]=basic_data_dataframe

    if(column.isnull().sum() > 0):
        nulls_metrics_dataframe=pd.DataFrame([get_null_info(df,column_name,target_aware)])
        dict_to_return["missings_metrics"]=nulls_metrics_dataframe

    dict_to_return["distribution_metrics"]=distribution_metrics_dataframe

    return dict_to_return


def get_null_info(df: pd.DataFrame, column_name, target_aware: bool) -> dict:
    nulls_info={}
    
    column=df[column_name]

    column_null_maks=column.isnull()
    null_total=column_null_maks.sum()
    null_porcentaje= column_null_maks.mean() * 100
    rows_with_null=df[column_null_maks]


    non_null_maks=~column.isnull()
    non_null_total=non_null_maks.sum()
    non_null_porcentaje=non_null_maks.mean() * 100
    non_null_values=df[non_null_maks]



    nulls_info["nulls_amount"]=null_total
    nulls_info["nulls_porcentaje"]=null_porcentaje

    nulls_info["non_null_amount"] = non_null_total
    nulls_info["non_null_porcentaje"] = non_null_porcentaje

    if(target_aware):
        target_correlation_nulls=rows_with_null["TARGET"].mean() * 100
        nulls_info["default_ratio_nulls"]=target_correlation_nulls

        default_ratio_non_null= non_null_values["TARGET"].mean() * 100
        nulls_info["non_null_default_ratio"] = default_ratio_non_null
    


    return nulls_info





def get_distribution_metrics(df: pd.DataFrame, column_name) -> dict:
    distribution_dict={}
    column=df[column_name]

    median= column.median()
    percentil_99=column.quantile(0.99)
    percentil_90=column.quantile(0.90)

    distribution_dict["skew"]=column.skew()
    distribution_dict["p90"]=percentil_90
    distribution_dict["p99"]=percentil_99

    if(median != 0):
        distribution_dict["ratio_p99_p50"]= percentil_99 / median
    if(percentil_90 !=0):
        distribution_dict["ratio_p99_p90"]= percentil_99 / percentil_90

    return distribution_dict
    

def basic_eda_per_column_categorical(df: pd.DataFrame,column_name, target_aware: bool) -> dict: 
    column=df[column_name]
    basic_data_dict={}
    cardinality=column.nunique(dropna=False)
    basic_data_dict["cardinality"]=cardinality
    basic_data_dict["mode"]=column.mode().to_list()
    dict_to_return={}
    dict_to_return["basic_data"]=pd.DataFrame([basic_data_dict])
    if(30 > cardinality and target_aware):
        default_rate_per_category=(df.groupby(column_name,dropna=False)["TARGET"].mean() *100).reset_index(name="TARGET_RATE") 
        dict_to_return["default_rate"]=default_rate_per_category
    dict_to_return["frequency"]=get_counts_per_class(column)
    return dict_to_return


def get_counts_per_class(column : pd.Series):
    value_count_serie=column.value_counts(dropna=False)
    cardinality=value_count_serie.shape[0]
    if(cardinality < 1000):
        result_df= value_count_serie.reset_index()
        result_df.columns= ["CATEGORY", "COUNT"]
        result_df["SEGMENT"] = "full"
        return result_df
    head_df= value_count_serie.head(20).reset_index()
    head_df.columns= ["CATEGORY", "COUNT"]
    head_df["SEGMENT"] = "top"
    tail_df= value_count_serie.tail(20).reset_index()
    tail_df.columns= ["CATEGORY", "COUNT"]
    tail_df["SEGMENT"] = "bottom"
    resume_df=pd.concat([head_df,tail_df],ignore_index=True)
    return resume_df


