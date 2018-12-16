# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:15:35 2018

@author: nadare
"""
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

######################################
# フォントがある場合のみ
import matplotlib as mpl
font = {"family":"Noto Sans CJK JP"}
mpl.rc('font', **font)
######################################

# space = spaces[1104]

ext_total_df = pd.read_hdf("../final_processed/newstadium_oldcount.h5", "data")
obj_col = ext_total_df.select_dtypes(["object", np.datetime64]).columns.tolist()
del_col = []

train_df = ext_total_df.query("1994 <= match_date_year <= 2016")
train_df.loc[:, "attendance_ratio"] = train_df["attendance"] / train_df["capacity"]
train_df.drop(ext_total_df.query("attendance <= 100").index, inplace=True)
test_df = ext_total_df.query("(division==1)&((match_date_year==2017)|((match_date_year==2018) & ((section<=17)|(33<=section))))").sort_values("id")

dropcol = ["id", "attendance", "attendance_ratio"]  + obj_col + del_col

#all_train_X = train_df.query("match_date_year <= 2016").drop(dropcol, axis=1)
#all_train_y = np.log1p(train_df.query("match_date_year <= 2016")["attendance"])
#all_train_y2 = train_df.query("match_date_year <= 2016")["attendance"] / train_df.query("match_date_year <= 2016")["capacity"]

test_X = test_df.drop(["id", "attendance"] + obj_col + del_col, axis=1)

smpsb_df = pd.read_csv("../input/sample_submit.csv", header=None)
smpsb_df.iloc[:, 1] = 0

with open("models.pickle", mode="rb") as f:
    gbms = pickle.load(f)

for i in tqdm(range(50)):
    gbm = gbms[i]

    test_pred = gbm.predict(test_X) * test_X["capacity"].values
    smpsb_df.iloc[:, 1] += test_pred / 50


smpsb_df.to_csv("../final_output/newstadium_oldcount_prediict.csv", index=None, header=None)