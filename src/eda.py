from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def calculate_vif(X, thresh=5.0):
    variables = X.values
    for i in range(variables.shape[1]):
        print(X.columns[i], variance_inflation_factor(variables, i))


def create_feature(data):
    df = data.copy()
    # df = df.drop(["area"], axis=1)
    df["walk_time_bin"] = pd.qcut(df["walk_time"], 10)
    df["frontage_size_bin"] = pd.qcut(df["frontage_size"], 10)
    print("dummy", df.select_dtypes(include=[object]).columns)
    df = pd.get_dummies(
        df,
        columns=list(df.select_dtypes(include=[object]).columns)
        + ["walk_time_bin", "frontage_size_bin"],
        drop_first=True,
    )
    bool_cols = df.select_dtypes(include=[bool]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    # standardize
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.columns = df.columns.str.replace("_(", "_").str.replace(", ", "-").str.replace("]", "")
    return df.drop(
        ["total_price", "price_per_tsubo", "walk_time", "frontage_size"],
        axis=1,
    )


def score(y, y_hat):
    mae = mean_absolute_error(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    corr = np.corrcoef(y, y_hat, rowvar=False)[0, 1]
    r2 = r2_score(y, y_hat)
    mape = np.median(np.abs((y - y_hat) / y)) * 100

    print(f"MAE\t: {mae}")
    print(f"RMSE\t: {rmse}")
    print(f"CORR\t: {corr}")
    print(f"R2\t: {r2}")
    print(f"MAPE\t: {mape}%")

    plt.figure(figsize=[8, 6], dpi=200)
    plt.scatter(y, y_hat, alpha=0.5)
    plt.legend()
    plt.show()

    return pd.DataFrame(np.array([[mae, rmse, corr, r2]]), columns=["mae", "rmse", "corr", "r2"])


def heatmap(coef, X_validation=None, scale=None):
    plt.figure(figsize=[8, 6], dpi=200)
    print(
        pd.Series(
            coef,
            index=X_validation.columns,
        )
    )
    if scale is None:
        sns.heatmap(coef.reshape(X_validation.shape[1], -1), center=0, cmap="bwr")
        plt.title("各パラメータ $\omega_i$ の推定値")
    else:
        sns.heatmap(
            coef.reshape(X_validation.shape[1], -1), center=0, cmap="bwr", vmin=-scale, vmax=scale
        )
        plt.title(f"各パラメータ $\omega_i$ の推定値：色調範囲[{-scale}, {scale}]")

    plt.yticks(np.arange(X_validation.shape[1]) + 0.5, X_validation.columns, rotation=0, fontsize=5)
    plt.ylim(X_validation.shape[1], 0)
    plt.grid()
    plt.show()
