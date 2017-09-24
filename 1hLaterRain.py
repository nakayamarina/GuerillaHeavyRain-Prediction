# ---------------------------------------------------------------------
# start    : 2017/09/24-
# filename : 1hLaterRain.py
# args     : 第1引数：雨量
#            第2引数：気温
#            第3引数：風速
#            第4引数：湿度
#            第5引数：1時間前と現在の気温の差
#            第6引数：1時間前と現在の風速の差
#            第7引数：1時間前と現在の湿度の差
# memo     : 保存した学習モデル(./model/)を使って1時間後の雨量を予測
# ---------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
import os.path


#保存した機械学習モデルを使って予測値を算出
def loaded_mod(mod_name, MU, SE):

    #保存した機械学習モデルをロード
    loaded_mod = pickle.load(open(mod_name, 'rb'))

    #予測値算出
    rain_zscore = loaded_mod.predict(wether)

    #算出された値はZ-scoreなので値を元に戻す
    #雨量のZ-score(rain_zscore) = 予測雨量(rain_later) - 教師データの平均雨量(MU) / 教師データの標準偏差(SE)より
    rain_later = (rain_zscore * SE) + MU

    print('======1時間後の雨量======')
    print('         %.3f          ' % (rain_later))
    print('=========================')
    print('by ' + mod_name + '\n')


if __name__ == '__main__':

    #気象情報を入力を受け取る
    rain = float(input("雨量："))
    temp = float(input("気温："))
    wind = float(input("風速："))
    humidity = float(input("湿度："))
    temp_ago = float(input("1時間前の気温："))
    wind_ago = float(input("1時間前の風速："))
    humidity_ago = float(input("1時間前の湿度："))
    print('\n')

    #ベクトル化（1次元配列だとWarningが出るので2次元配列にしておく）
    wether = np.array([[rain, temp, wind, humidity, (temp_ago - temp), (wind_ago - wind), (humidity_ago - humidity)]])

    #入力データの標準化
    #教師データ読み込み
    data_train = pd.read_csv('./kochi_train/input.csv')
    data_train = data_train.iloc[:, (len(data_train.columns)-7):len(data_train.columns)]

    #教師データの各気象情報の平均、標準偏差算出
    MU = list(data_train.mean())
    SE = list(data_train.std())

    #教師データの平均、標準偏差を使って入力データを標準化
    for i in range(7):
        wether[:, [i]] = (wether[:, [i]] - MU[i]) / SE[i]


    #機械学習モデルをロード
    #保存した機械学習モデルのファイル名を読み込む
    models = os.listdir('./model/')

    for i in range(len(models)):
        mod_name = './model/' + models[i]
        #保存した機械学習モデルを使って予測値を算出
        loaded_mod(mod_name, MU[0], SE[0])
