# ---------------------------------------------------------------------
# start    : 2017/09/15-
# filename : wether2csv.py
# data     : 気象庁>過去の気象データ>高知県南国し後免からDLした
#            気温、湿度、風向・風量、雨量のcsvファイル
# memo     : DLした気象データを整形・結合・csv書き出し
# coution  : 気象・土地によってcsvの中身が変わるので整形の際は中身を確認してから
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd

#各気象データ読み込み

#年月日時、数値だけに整形、属性名を変更
#0行目に関係のないものが入っている場合はdropで削除

#雨量
rain_data = pd.read_csv("./dataset_kochi/rain.csv", encoding = 'shift-jis', skiprows = 3)

rain = rain_data.iloc[:, 0:2]
rain.columns = ['date', 'rain']

#気温
temp_data = pd.read_csv("./dataset_kochi/temp.csv", encoding = 'shift-jis', skiprows = 1)

temp_data.columns = ['date', 'temp']
temp = temp_data.drop(0)

#風向・風量
wind_data = pd.read_csv("./dataset_kochi/wind.csv", encoding = 'shift-jis', skiprows = 3)

wind = wind_data.iloc[:, 0:2]
wind.columns = ['date', 'wind']
wind = wind.drop(0)

#湿度
humidity_data = pd.read_csv("./dataset_kochi/humidity.csv", encoding = 'shift-jis', skiprows = 1)

humidity_data.columns = ['date', 'humidity']
humidity = humidity_data.drop(0)


#整形した各気象データを日付に対応させて連結(merge)する
temp_df = pd.merge(rain, temp, on='date')
temp_df = pd.merge(temp_df, wind, on='date')
temp_df = pd.merge(temp_df, humidity, on='date')

#整形データをcsvで書き出し
temp_df.to_csv('wetherInfo.csv')
