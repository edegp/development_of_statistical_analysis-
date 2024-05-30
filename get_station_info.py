from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
from sklearn.calibration import LabelEncoder

from src.consts import data_path, data_file
from src.station import station_to_coords

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_path, data_file), encoding="cp932")

    with ThreadPoolExecutor(max_workers=4) as executor:
        stations = list(executor.map(station_to_coords, df["最寄駅：名称"].unique()))

    station_df = pd.DataFrame(
        {k: [station[k] for station in stations if station is not None] for k in stations[0].keys()}
    )
    station_df.to_csv(os.path.join(data_path, "stations.csv"), index=False)
