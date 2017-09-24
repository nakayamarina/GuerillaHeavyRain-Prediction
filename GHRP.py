# ---------------------------------------------------------------------
# start    : 2017/09/16-
# filename : GHRPrediction.py
# args     : 第1引数：教師用データに使用するcsvファイルのあるディレクトリ
#            第2引数：テスト用データに使用するcsvファイルへのあるディレクトリ
# data     : ./kochi_train/wetherInfo.csv
#            wether_train.pyで書き出したcsvファイル
# memo     : 天候情報を学習・ゲリラ豪雨がくるかどうか識別率算出
# ---------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# 平均二乗誤差を評価するためのメソッドを呼び出し
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pickle
import numpy as np
import pandas as pd
import sys

#csv読み込み、欠損値除去
def input_data(dr):
    #読み込みたいcsvファイルへのパス作成、読み込み
    path = dr + 'wetherInfo.csv'
    input_data = pd.read_csv(path)

    #必要な部分だけ抽出
    input_data = input_data.iloc[:, (len(input_data.columns)-4):len(input_data.columns)]

    #欠損値除去
    input_data = input_data.dropna()

    return input_data


#特徴抽出、ベクトル化

#特徴追加
def add_data(data):
    #1時間前との差分を追加
    data['temp_dfr'] = data['temp'] - data['temp'].shift(1)
    data['wind_dfr'] = data['wind'] - data['wind'].shift(1)
    data['humidity_dfr'] = data['humidity'] - data['humidity'].shift(1)

    #欠損値除去
    data = data.dropna()

    return data

#標準化、ベクトル化
def train_vec(data_train):

    #標準化
    zscore_data = data_train.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)

    #ベクトル化
    x_data = zscore_data.as_matrix()

    #用いた特徴csv書き出し
    zscore_data.to_csv('./kochi_train/zscore.csv')

    return x_data, zscore_data

def test_vec(data_test, data_train):

    #教師データの平均・偏差値を使って標準化
    MU = list(data_train.mean())
    SE = list(data_train.std())

    zscore_data = pd.DataFrame(columns=['rain', 'temp', 'wind', 'humidity', 'temp_dfr', 'wind_dfr', 'humidity_dfr'])

    for i in range(7):
        zscore_data.iloc[:, i] = (data_test.iloc[:, i] - MU[i]) / SE[i]

    #ベクトル化
    x_data = zscore_data.as_matrix()

    #用いた特徴csv書き出し
    zscore_data.to_csv('./kochi_test2016/zscore.csv')

    return x_data, zscore_data



#ラベル作成：Neural Network
def MLPC_label_vec(label_vec):
    #1時間後に雨が10mm以上降れば１、降らなければ０でラベルを作成
    label_vec = np.array(label_vec.rain >= 10, dtype = 'int')
    y_data = np.roll(label_vec, -1)
    y_data[len(y_data) - 1] = 0

    return y_data


#学習・識別率算出：Neural Network
def MLPC(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    clf = MLPClassifier(hidden_layer_sizes=(1000, 500, 100), random_state=1)
    clf.fit(x_train, y_train)

    # モデルを保存する
    pickle.dump(clf, open('./model/NeuralNetwork.sav', 'wb'))

    #識別率算出
    predicted = clf.predict(x_test)
    print('MLP')
    print(metrics.classification_report(y_test, predicted))


#ラベル作成：Regression
def R_label_vec(input_data):
    #1時間後の雨量
    data = np.array(input_data.rain)
    label_vec = np.roll(data, -1)
    label_vec[len(label_vec)-1] = 0

    return label_vec


#学習・識別率算出：Linear Regression
def LR(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    mod = LinearRegression(fit_intercept = True, normalize = True, copy_X = True, n_jobs = 1)
    mod.fit(x_train, y_train)

    # モデルを保存する
    pickle.dump(mod, open('./model/LinerRegression.sav', 'wb'))

    #作成したモデルから予測
    y_train_pred = mod.predict(x_train)
    y_test_pred = mod.predict(x_test)

    #識別率算出
    print('Liner Regression: %.3f' % (mod.score(x_test,y_test)))

    #教師用、テスト用データに関して平均二乗誤差を出力
    #小さいほどモデルの性能がいい
    print('Liner Regression MSE Train : %.3f, Test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    #教師用、テスト用データに関してR^2を出力
    #1に近いほどモデルの性能がいい
    print('Liner Regression R^2 Train : %.3f, Test : %.3f' % (mod.score(x_train, y_train), mod.score(x_test, y_test)))


#学習・識別率算出：Neural Network Regression
def MLPR(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    mod = MLPRegressor(hidden_layer_sizes=(1000, 500,100,),random_state=42)
    mod.fit(x_train, y_train)

    # モデルを保存する
    pickle.dump(mod, open('./model/NeuralNetworkRegression.sav', 'wb'))

    #識別率算出
    print('Neural Network Regression: %.3f' % (mod.score(x_test,y_test)))


#学習・識別率算出：Random Forest Regression
def RFR(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)

    # モデルを保存する
    pickle.dump(forest, open('./model/RandomForestRegression.sav', 'wb'))

    #作成したモデルから予測
    y_train_pred = forest.predict(x_train)
    y_test_pred = forest.predict(x_test)

    #識別率算出
    print('Random Forest Regression: %.3f' % (forest.score(x_test,y_test)))

    #教師用、テスト用データに関して平均二乗誤差を出力
    #小さいほどモデルの性能がいい
    print('Random Forest Regression MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )

    #教師用、テスト用データに関してR^2を出力
    #1に近いほどモデルの性能がいい
    print('Random Forest Regression R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )



#学習・識別率算出：Logistic Regression
def LogisticR(x_train, y_train, x_test, y_test):
    #オブジェクト生成
    mod = LogisticRegression()
    mod.fit(x_train, y_train)

    # モデルを保存する
    pickle.dump(mod, open('./model/RogisticRegression.sav', 'wb'))    

    #作成したモデルから予測
    y_train_pred = mod.predict(x_train)
    y_test_pred = mod.predict(x_test)

    #識別率算出
    print('Logistic Regression: %.3f' % (mod.score(x_test,y_test)))

    #教師用、テスト用データに関して平均二乗誤差を出力
    #小さいほどモデルの性能がいい
    print('Logistic Regression MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )

    #教師用、テスト用データに関してR^2を出力
    #1に近いほどモデルの性能がいい
    print('Logistic Regression R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )


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

    #特徴抽出、ベクトル化
    data_train = add_data(input_train)
    data_test = add_data(input_test)
    data_train.to_csv('./kochi_train/input.csv')
    data_test.to_csv('./kochi_test2016/input.csv')

    x_train, zscore_train = train_vec(data_train)
    x_test, zscore_test = test_vec(data_test, data_train)

    #ラベル作成：Neral Network
    ny_train = MLPC_label_vec(data_train)
    ny_test = MLPC_label_vec(data_test)

    #学習・識別率算出：Neural Network
    MLPC(x_train, ny_train, x_test, ny_test)

    #ラベル作成：Regression
    y_train = R_label_vec(zscore_train)
    y_test = R_label_vec(zscore_test)

    #学習・識別率算出：Linear Regression
    LR(x_train, y_train, x_test, y_test)

    #学習・識別率算出：Neural Network Regression
    #MLPR(x_train, y_train, x_test, y_test)

    #学習・識別率算出：Random Forest Regression
    RFR(x_train, y_train, x_test, y_test)

    #学習・識別率算出：
    LogisticR(x_train, ny_train, x_test, ny_test)
