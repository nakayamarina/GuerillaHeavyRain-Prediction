# ---------------------------------------------------------------------
# start    : 2017/09/16-
# filename : GHRPrediction.py
# args     : 第1引数：教師用データに使用するcsvファイルのあるディレクトリ
#            第2引数：テスト用データに使用するcsvファイルへのあるディレクトリ
# data     : ./wetherInfo.csv
#            wether2csv.pyで書き出したcsvファイル
# memo     : 雨量10mm以上をゲリラ豪雨とし、NNで学習・識別率算出
# ---------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression

# 平均二乗誤差を評価するためのメソッドを呼び出し
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import sys

def input_data(dr):
    #読み込みたいcsvファイルへのパス作成、読み込み
    path = dr + 'wetherInfo.csv'
    input_data = pd.read_csv(path)
    #欠損地除去
    input_data = input_data.dropna()

    return input_data

def data_vec(input_data):
    data = input_data.iloc[:, 2:6]
    #ベクトル化
    data_vec = data.as_matrix()

    return (data_vec)


#Neural Network
def NN_label_vec(input_data):
    #1時間後に雨が10mm以上降れば１、降らなければ０でラベルを作成
    data = np.array(input_data.rain >= 5, dtype = 'int')
    label_vec = np.roll(data, -1)
    label_vec[len(label_vec) - 1] = 0

    return label_vec

def NN(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=1)
    clf.fit(x_train, y_train)

    #識別率算出
    predicted = clf.predict(x_test)
    print(metrics.classification_report(y_test, predicted))


#Liner Regression
def LR_label_vec(input_data):
    data = np.array(input_data.rain)
    label_vec = np.roll(data, -1)
    label_vec[len(label_vec)-1] = 0

    return label_vec

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

    #ベクトル化
    x_train = data_vec(input_train)
    x_test = data_vec(input_test)



    #ラベル作成

    #Neral Network
    #y_train = NN_label_vec(input_train)
    #y_test = NN_label_vec(input_test)

    #Liner Regression
    y_train = LR_label_vec(input_train)
    y_test = LR_label_vec(input_test)

    #学習・識別率算出

    #Neural Network
    #NN(x_train, y_train, x_test, y_test)

    #Liner Regression
    LR(x_train, y_train, x_test, y_test)
