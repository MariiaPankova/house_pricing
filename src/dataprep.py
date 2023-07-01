import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    cross_val_score,
    RepeatedKFold,
)
from dvclive import Live
from dvc.api import params_show
from rich import print
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

RANDOM_STATE = 42


def df_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    yn_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    for col in yn_cols:
        df[col].replace("yes", 1, inplace=True)
        df[col].replace("no", 0, inplace=True)

    encoder = OneHotEncoder(handle_unknown="ignore")
    enc_df = pd.DataFrame(
        encoder.fit_transform(df[["furnishingstatus"]]).toarray(),
        columns=df["furnishingstatus"].unique(),
    )
    transformed_df = df.join(enc_df)
    transformed_df.drop("furnishingstatus", axis=1, inplace=True)
    X = transformed_df.drop(["price"], axis="columns")
    Y = transformed_df["price"]
    return X, Y


def main(csv_path: str = "data/Housing.csv"):
    model_params = params_show()
    model = XGBRegressor(**model_params)
    X, Y = df_prepare(csv_path)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        Y,
        scoring={"acc": "accuracy", "presision": "precision", "rec": "recall"},
        cv=cv,
        n_jobs=-1,
    )
    with Live(save_dvc_exp=True) as live:
        for k, v in scores.items():
            live.log_metric(f"{k}_mean", v.mean())
            live.log_metric(f"{k}_std", v.std())


if __name__ == "__main__": 
    main()
