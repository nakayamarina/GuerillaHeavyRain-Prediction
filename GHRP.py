# ---------------------------------------------------------------------
# start    : 2017/09/16-
# filename : GHRPrediction.py
# args     : 第1引数：教師用データに使用するcsvファイルのあるディレクトリ
#            第2引数：テスト用データに使用するcsvファイルへのあるディレクトリ
# data     : ./wetherInfo.csv
#            wether2csv.pyで書き出したcsvファイル
# memo     : 雨量10mm以上をゲリラ豪雨とし、学習・識別率算出
#            手法：Neural Network, LinearRegression
# ---------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression

# 平均二乗誤差を評価するためのメソッドを呼び出し
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import sys

#csv読み込み、欠損値除去
def input_data(dr):
    #読み込みたいcsvファイルへのパス作成、読み込み
    path = dr + 'wetherInfo.csv'
    input_data = pd.read_csv(path)
    #欠損地除去
    input_data = input_data.dropna()

    return input_data


#特徴抽出
def data_fe(data):
    data['temp_dfr'] = data['temp'] - data['temp'].shift(1)
    data['wind_dfr'] = data['wind'] - data['wind'].shift(1)
    data['humidity_dfr'] = data['humidity'] - data['humidity'].shift(1)
    data = data.dropna()

    return data


#ベクトル化
def data_vec(data_vec):
    data_vec = data_vec.iloc[:, 2:6]
    #ベクトル化
    x_data = data_vec.as_matrix()

    return x_data


#ラベル作成：Neural Network
def NN_label_vec(label_vec):
    #1時間後に雨が10mm以上降れば１、降らなければ０でラベルを作成
    label_vec = np.array(label_vec.rain >= 5, dtype = 'int')
    y_data = np.roll(data, -1)
    y_data[len(y_data) - 1] = 0

    return y_data


#学習・識別率算出：Neural Network
def NN(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=1)
    clf.fit(x_train, y_train)

    #識別率算出
    predicted = clf.predict(x_test)
    print(metrics.classification_report(y_test, predicted))


#ラベル作成：Linear Regression
def LR_label_vec(input_data):
    data = np.array(input_data.rain)
    label_vec = np.roll(data, -1)
    label_vec[len(Neural Networklabel_vec)-1] = 0

    return label_vec


#学習・識別率算出：Linear Regression
def LR(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    mod = LinearRegression(fit_intercept = True, normalize = True, copy_X = True, n_jobs = 1)
    #教師用データでパラメータ推定
    mod.fit(x_train, y_train)

    #作成したモデルから予測
    y_train_pred = mod.predict(x_train)
    y_test_pred = mod.predict(x_test)

    #教師用、テスト用データに関して平均二乗誤差を出力
    #小さいほどモデルの性能がいい
    print('MSE Train : %.3f, Test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    #教師用、テスト用データに関してR^2を出力
    #1に近いほどモデルの性能がいい
    print('R^2 Train : %.3f, Test : %.3f' % (mod.score(x_train, y_train), mod.score(x_test, y_test)))



if __name__ == '__main__':

    #train -> 教師用
    #test -> テスト用

    #読み込みたいcsvファイルのあるディレクトを引数として取得
    args = sys.argv
    dr_train = args[1]
    dr_test = args[2]

    #csv読み込み、欠損地除去
    input_train = input_data(dr_train)
    input_test = input_data(dr_test)

    #特徴抽出
    data_train = data_fe(input_train)
    data_test = data_fe(input_test)

    #ベクトル化
    x_train = data_vec(data_train)
    x_test = data_vec(data_test)


    #ラベル作成：Neral Network
    #y_train = NN_label_vec(data_train)
    #y_test = NN_label_vec(data_test)

    #学習・識別率算出：Neural Network
    #NN(x_train, y_train, x_test, y_test)


    #ラベル作成：Linear Regression
    y_train = LR_label_vec(input_train)
    y_test = LR_label_vec(input_test)

    #学習・識別率算出：Linear Regression
    LR(x_train, y_train, x_test, y_test)
