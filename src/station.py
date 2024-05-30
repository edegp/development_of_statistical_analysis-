import os
import requests
import numpy as np
from time import sleep

from src.consts import STATION_API_BASE


def station_to_coords(x):
    sleep(1)
    if type(x) == float and np.isnan(x):
        return None

    x = x.replace("(東京)", "")
    if x in ["祖師ケ谷大蔵", "梅ケ丘", "富士見ケ丘", "桜ケ丘"]:
        x = x.replace("ケ", "ヶ").replace("が", "ヶ")

    url = os.path.join(STATION_API_BASE, "json?method=getStations")
    params = {
        "name": x,
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        res_json = res.json()
        stations = [
            station
            for station in res_json["response"]["station"]
            if station["prefecture"] == "東京都"
        ]
        if len(res_json["response"]["station"]) > 1:
            print(x, res_json["response"], stations)
        return stations[0]
    except Exception as e:
        print(e, x, res.json())
        return None
