import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import (
    cross_validate,
    RepeatedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dvclive import Live
from dvc.api import params_show
from rich import print
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_STATE = 42
sns.set_style()


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


def get_model(model_params: dict):
    yn_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    pipeline = Pipeline(
        [
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        *(
                            (f"{col}-binalizer", OrdinalEncoder(), [col])
                            for col in yn_cols
                        ),
                        ("One-hot-furniture", OneHotEncoder(), ["furnishingstatus"]),
                    ],
                    remainder="passthrough",
                ),
            ),
            (
                "estimator",
                XGBRegressor(**model_params, random_state=RANDOM_STATE),
            ),
        ]
    )
    return pipeline


def main(csv_path: str = "data/Housing.csv"):
    params = params_show()
    model = get_model(params["model"])
    print(model)
    housing_data = pd.read_csv(csv_path)
    X = housing_data.drop(["price"], axis="columns")
    Y = MinMaxScaler().fit_transform(housing_data[["price"]])

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        Y,
        scoring={"neg_mse": "neg_root_mean_squared_error", "r2": "r2"},
        cv=cv,
        n_jobs=-1,
    )

    sns.boxplot(scores["test_neg_mse"] * (-1)).set(xlabel="MSE")
    plt.savefig("dvclive/plots/mse.png")
    plt.clf()
    sns.boxplot(scores["test_r2"]).set(xlabel="$r^2$")
    plt.savefig("dvclive/plots/r2.png")

    print(scores)

    with Live(save_dvc_exp=True) as live:
        for k, v in scores.items():
            live.log_metric(f"{k}_mean", v.mean())
            print(f"{k}_mean", v.mean())


if __name__ == "__main__":
    main()
