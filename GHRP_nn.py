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


def label_vec(input_data):
    #1時間後に雨が10mm以上降れば１、降らなければ０でラベルを作成
    data = np.array(input_data.rain >= 5, dtype = 'int')
    label_vec = np.roll(data, -1)
    label_vec[len(label_vec) - 1] = 0
    
    return label_vec


def NN(train_vec, train_lab, test_vec, test_lab):
    #MLPClassifier適用
    clf = MLPClassifier(hidden_layer_sizes=(100,100), random_state=1)
    clf.fit(train_vec, train_lab)

    #clf（トレーニング済み）にテスト用データを適用
    predicted = clf.predict(test_vec)
    print(metrics.classification_report(test_lab, predicted))
    
    

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
    train_vec = data_vec(input_train)
    test_vec = data_vec(input_test)
    
    #ラベル作成
    train_lab = label_vec(input_train)
    test_lab = label_vec(input_test)

    #学習・識別率算出
    NN(train_vec, train_lab, test_vec, test_lab)


