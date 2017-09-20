# ---------------------------------------------------------------------
# start    : 2017/09/20-
# filename : wether_train.py
# args     : 第1引数：下記の共通部分　例）kochi_train
#            読み込みたいcsvファイルがあるディレクトリ 例）kochi_train2013, kochi_train2014
#            保存先のディレクトリ 例）kochi_train
# data     : wetherInfo.csv
# memo     : wetherInfo.csvの縦結合（教師データを増やす）
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import os.path
import sys

#読み込みたいcsvファイルがあるディレクトリ、保存先のディレクトリ名の共通部分
args = sys.argv
dr_name = args[1]

#path内にあるディレクトリの名前取得
path = './'
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]


#dr_nameが含まれるディレクトリ名だけ取得 -> train_dir
train_dir = []

for i in range(0,len(files_dir)):
        if(dr_name in files_dir[i]):
            train_dir.append(files_dir[i])

#wetherInfo.csvがあれば読み込み、縦に結合
count = 0
for i in range(0, len(train_dir)):
    wetherInfo = './' + train_dir[i] + '/' + 'wetherInfo.csv'

    if(os.path.exists(wetherInfo) and count == 0):
        train = pd.read_csv(wetherInfo)
        count += 1

    elif(os.path.exists(wetherInfo) and count > 0):
        train_add = pd.read_csv(wetherInfo)
        train = pd.concat([train, train_add])

#csvで書き出し
output = './' + dr_name + '/' + 'wetherInfo.csv'
train.to_csv(output)
