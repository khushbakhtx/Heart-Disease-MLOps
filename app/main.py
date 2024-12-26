import numpy as np
import pandas as pd
import optuna
import catboost as cb
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuna.visualization import plot_parallel_coordinate, plot_optimization_history
from preprocessing.data_preprocessing import prepare_data
from utils.utils import plot_feature_importance, plot_roc_curve

RANDOM_STATE = 42
TEST_SIZE = 0.3

EXPERIMENT_NAME = 'first_experiment'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

def objective(trial):
    df = pd.read_csv("data/data_wo_processing.csv")
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.3, random_state=42)

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "3gb",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)

    return accuracy

if __name__ == "__main__":
    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, timeout=600)

        trial = study.best_trial
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_accuracy", trial.value)

        fig_pc = plot_parallel_coordinate(study)
        fig_poh = plot_optimization_history(study)

        fig_pc.write_image("parallel_coordinate.png")
        fig_poh.write_image("optimization_history.png")

        mlflow.log_artifact("parallel_coordinate.png")
        mlflow.log_artifact("optimization_history.png")

        best_params = trial.params
        gbm = cb.CatBoostClassifier(**best_params)
        
        df = pd.read_csv("data/data_wo_processing.csv")
        target = df['HeartDisease']
        features = df.drop(columns=['HeartDisease'])
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)
        
        mlflow.sklearn.log_model(gbm, "catboost_model_modified")

        fig = plot_feature_importance(gbm)
        mlflow.log_figure(fig, "feature_importance.png")

        fig2 = plot_roc_curve(gbm, X_test, y_test)
        mlflow.log_figure(fig2, "roc_curve.png")