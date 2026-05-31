import numpy as np
import pandas as pd
from typing import Any
import sklearn as skt
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.base import BaseEstimator
import mlflow.sklearn
from sklearn.model_selection import BaseCrossValidator
from mlflow.models import infer_signature
from default_risk.config import ARTIFACTS_DIR


def run_cv_tracked_mlflow(model_class: type[BaseEstimator], model_params: dict[str, Any], cv: BaseCrossValidator, X: pd.DataFrame, Y: pd.Series, experiment_name: str, run_name: str,persist_feature_importance: bool =True,save_final_model: bool =True,enable_models_autlog: bool=False) -> None:

    mlflow.set_experiment(experiment_name)
    mlflow.xgboost.autolog(log_models=enable_models_autlog, silent=True)
    parent_name= "Parent_" + run_name
    oof_auc_prob=np.zeros(len(Y)) #for pre-alocated memory 
    list_scores=[]
    
    fold=0
    with mlflow.start_run(run_name= parent_name) as parent_run:
        mlflow.log_params(model_params)
        mlflow.log_param("n_folds",cv.get_n_splits())
        for train_index, val_index in cv.split(X,Y) : 
            fold = fold +1
            x_train= X.iloc[train_index]
            y_train= Y.iloc[train_index]
            x_val= X.iloc[val_index]
            y_val= Y.iloc[val_index]
            model= model_class(**model_params)
            name_of_nested_run= f"{run_name}_child_{fold:d}"

            with mlflow.start_run(run_name= name_of_nested_run, nested=True) :
                model.fit(x_train,y_train,eval_set =[(x_val,y_val)])
                predictions= model.predict_proba(x_val)
                default_prob= predictions[:,1]
                oof_auc_prob[val_index] = default_prob
                score= roc_auc_score(y_val,default_prob)
                list_scores.append(score)
                if(persist_feature_importance):
                    save_feature_importance(model,x_train)

        mean_auc= np.mean(list_scores)
        standar_deviation= np.std(list_scores)
        mlflow.log_metric("cv_mean_auc",(mean_auc))
        mlflow.log_metric("cv_mean_std",(standar_deviation))
        oof_auc_score = roc_auc_score(Y,oof_auc_prob)
        mlflow.log_metric("oof_auc",oof_auc_score)
        print("=================================================================================")
        print(f"result of CV with {fold:d} folds")
        print(f"mean AUC per fold= {mean_auc:.3f} ± {standar_deviation:.3f}(std)")
        print(f"auc_score_OOF= {oof_auc_score:.3f}")
        print("=================================================================================")
        if(save_final_model) :
            final_model=model_class(**model_params)
            final_model.fit(X,Y)
            input_sample = X.head(5)
            predict_input_sample= final_model.predict(input_sample)
            signature= infer_signature(input_sample,predict_input_sample)
            mlflow.sklearn.log_model(final_model, "final_model",signature=signature)
            save_feature_importance(final_model,X)
    return

def persist_and_export_feature_importance(feature_importance_df: pd.DataFrame)-> None:
    file_path= str(ARTIFACTS_DIR / "feature_importance.csv")
    with open(file_path, "w", encoding="utf-8") as f:
        feature_importance_df.to_csv(f,index=False)
    mlflow.log_artifact(file_path)
    return


def save_feature_importance(trained_model, X: pd.DataFrame) -> None:
    df_feature_importance=pd.DataFrame()
    features_names= X.columns
    importances= trained_model.feature_importances_
    df_feature_importance["feature_name"] = features_names
    df_feature_importance["importances"] = importances
    df_feature_importance.sort_values(by="importances",ascending=True,inplace=True)
    persist_and_export_feature_importance(df_feature_importance)
    return