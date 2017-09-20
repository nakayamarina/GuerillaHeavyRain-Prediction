# ---------------------------------------------------------------------
# start    : 2017/09/15-
# filename : wether2csv.py
# args     : 第1引数：読み込みたいcsvファイルのあるディレクトリ
# data     : ./dataset-kochi/
#            気象庁>過去の気象データ>高知県南国し後免からDLした
#            1年分の時間ごとの気温、湿度、風向・風量、雨量のcsvファイル
# memo     : DLした気象データを整形・結合・csv書き出し
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import sys

#読み込みたいcsvファイルのあるディレクトを引数として取得
args = sys.argv
dr = args[1]

#読み込みたいcsvファイルへのパス作成
rpath = dr + 'rain.csv'
tpath = dr + 'temp.csv'
wpath = dr + 'wind.csv'
hpath = dr + 'humidity.csv'

#各気象データ読み込み
#行頭数行に不要な行があるのでskiprowsで除外
#年月日時、数値だけに整形、属性名を変更

#雨量
rain_data = pd.read_csv(rpath, encoding = 'shift-jis', skiprows = 4)

rain = rain_data.iloc[:, 0:2]
rain.columns = ['date', 'rain']

#気温
temp_data = pd.read_csv(tpath, encoding = 'shift-jis', skiprows = 3)

temp_data.columns = ['date', 'temp']
temp = temp_data

#風向・風量
wind_data = pd.read_csv(wpath, encoding = 'shift-jis', skiprows = 4)

wind = wind_data.iloc[:, 0:2]
wind.columns = ['date', 'wind']

#湿度
humidity_data = pd.read_csv(hpath, encoding = 'shift-jis', skiprows = 3)

humidity_data.columns = ['date', 'humidity']
humidity = humidity_data


#整形した各気象データを日付に対応させて連結(merge)する
temp_df = pd.merge(rain, temp, on='date')
temp_df = pd.merge(temp_df, wind, on='date')
temp_df = pd.merge(temp_df, humidity, on='date')

#最後の行に2016年のデータがあるので削除
temp_df = temp_df.drop(len(temp_df)-1)

#整形データをcsvで書き出し
output = dr + 'wetherInfo.csv'
temp_df.to_csv(output)
