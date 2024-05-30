import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import statsmodels.api as sm
import lightgbm as lgb

from .consts import data_path


def convert_walk_minute(x):
    if type(x) != str:
        return x
    print(x)
    if "分" in x:
        return np.mean(list(map(int, x.replace("～", "").split("分")[:2])))
    if "H" in x:
        print(x)
        s = x.split("～")
        if "" in s:
            s.remove("")
        print(s)
        if s == [""]:
            return 20
        return np.mean(
            list(
                map(
                    lambda x: int(x.split("H")[0]) * 60
                    + (int(x.split("H")[1]) if x.split("H")[1] != "" else 0),
                    s,
                )
            )
        )
    return int(x)


def create_feature(raw):
    df = raw.copy()
    df["is_only_land"] = df["種類"] == "宅地(土地)"
    df["is_closed_price"] = df["価格情報区分"] == "成約価格情報"
    df["is_commercial_area"] = df["地域"] == "商業地"
    df["walk_time"] = df["最寄駅：距離（分）"].apply(convert_walk_minute)
    df["building_age"] = (
        df["建築年"].str.replace("年", "").str.replace("戦前", "1945").astype(float)
    )
    # 大きさ
    print(df["間口"].str.replace("m以上", "").unique())
    df["frontage_size"] = df["間口"].str.replace("m以上", "").astype(float)
    print(df["frontage_size"].unique())
    df["land_size"] = df["面積（㎡）"].str.replace("㎡以上|,", "", regex=True).astype(float)
    df["building_size"] = df["延床面積（㎡）"].str.replace("㎡以上|,", "", regex=True).astype(float)

    # 　特徴
    df["is_bag_fabric"] = df["土地の形状"] == "袋地等"
    df["is_irregular_shape"] = df["土地の形状"] == "不整形"
    df["is_steel"] = df["建物の構造"].str.contains("鉄骨", na=False)
    df["is_rc"] = df["建物の構造"].str.contains("ＲＣ", na=False)
    df["is_wood"] = df["建物の構造"].str.contains("木造", na=False)
    df["is_block"] = df["建物の構造"].str.contains("ブロック", na=False)

    df["is_used_for_store"] = df["用途"].str.contains("店舗", na=False)
    df["is_used_for_office"] = df["用途"].str.contains("事務所", na=False)
    df["is_used_for_factory"] = df["用途"].str.contains("工場", na=False)

    df["is_only_residential_area"] = df["都市計画"].str.contains("住専", na=False)
    df["is_residential_area"] = df["都市計画"].str.contains("住居", na=False)

    df["is_purpose_for_living"] = df["今後の利用目的"].str.contains("住居", na=False)
    df["is_no_front_road"] = df["前面道路：方位"] == "接面道路無"
    df["is_front_of_large_road"] = (df["前面道路：種類"] == "都道") | (
        df["前面道路：種類"] == "国道"
    )

    df["is_dealing_as_competition"] = df["取引の事情等"].str.contains("調停・競売", na=False)

    df["term"] = df["取引時期"].str.replace("年第|四半期", "", regex=True).astype(int)

    df["coverage_ratio"] = df["建ぺい率（％）"].fillna(df["建ぺい率（％）"].mode().values[0])
    df["floor_area_ratio"] = df["容積率（％）"].fillna(df["容積率（％）"].mode().values[0])
    df["walk_time"] = df["walk_time"].fillna(np.mean(df["walk_time"]))
    df["building_age"] = df["building_age"].fillna(np.mean(df["building_age"]))
    df["frontage_size"] = df["frontage_size"].fillna(np.mean(df["frontage_size"]))
    df["land_size"] = df["land_size"].fillna(np.mean(df["land_size"]))
    df["building_size"] = df["building_size"].fillna(np.mean(df["building_size"]))
    df["term"] = df["term"].fillna(np.mean(df["term"]))
    df["front_load_width"] = df["前面道路：幅員（ｍ）"].fillna(np.mean(df["前面道路：幅員（ｍ）"]))

    df["price_per_tsubo"] = df["坪単価"].fillna(0)
    df["price_per_square"] = df["取引価格（㎡単価）"].fillna(0)

    station_df = pd.read_csv(os.path.join(data_path, "stations.csv"))
    df = pd.merge(
        df,
        station_df[["name", "x", "y", "line"]].rename({"name": "最寄駅：名称"}, axis=1),
        on="最寄駅：名称",
        how="left",
    )

    df = df.drop(
        [
            "市区町村コード",
            "都道府県名",
            "市区町村名",
            "種類",
            "価格情報区分",
            "地域",
            "最寄駅：距離（分）",
            "建物の構造",
            "面積（㎡）",
            "延床面積（㎡）",
            "間口",
            "土地の形状",
            "建物の構造",
            "用途",
            "都市計画",
            "今後の利用目的",
            "建築年",
            "前面道路：方位",
            "前面道路：種類",
            "取引の事情等",
            "取引時期",
            "最寄駅：名称",
            "建ぺい率（％）",
            "容積率（％）",
            "前面道路：幅員（ｍ）",
            "坪単価",
            "取引価格（㎡単価）",
        ],
        axis=1,
    )

    le = LabelEncoder()
    df["line_label"] = le.fit_transform(df["line"])
    area = df["地区名"]
    df = pd.get_dummies(df, columns=["地区名"], drop_first=True)

    train_df = df[df["x"].notnull()].copy()

    X = train_df[
        ["walk_time"] + [col for col in train_df.columns if col.startswith("地区名_")]
    ].astype(float)
    x_dataset = lgb.Dataset(X, train_df["x"].values)
    x_model = lgb.train(train_set=x_dataset, params={"objective": "regression"})
    y_dataset = lgb.Dataset(X, train_df["y"].values)
    y_model = lgb.train(train_set=y_dataset, params={"objective": "regression"})
    line_dataset = lgb.Dataset(X, train_df["line_label"].values)
    line_model = lgb.train(
        train_set=line_dataset, params={"objective": "multiclass", "num_class": len(le.classes_)}
    )

    test_df = df[df["x"].isnull()].copy()
    X = test_df[
        ["walk_time"] + [col for col in test_df.columns if col.startswith("地区名_")]
    ].astype(float)
    test_df["x"] = x_model.predict(X)
    test_df["y"] = y_model.predict(X)
    test_df["line_label"] = np.argmax(line_model.predict(X), axis=1)

    df = pd.concat([train_df, test_df], axis=0)
    df["line"] = le.inverse_transform(df["line_label"])
    df = df.drop(
        ["line_label"] + [col for col in test_df.columns if col.startswith("地区名_")], axis=1
    )
    df["area"] = area

    df = df.rename(
        {
            "地区名": "area",
            "取引価格（総額）": "total_price",
        },
        axis=1,
    )

    return df
