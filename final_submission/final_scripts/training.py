# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:15:35 2018

@author: nadare
"""
import pickle
import warnings
warnings.filterwarnings("ignore")

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

all_train_X = train_df.query("match_date_year <= 2016").drop(dropcol, axis=1)
all_train_y = np.log1p(train_df.query("match_date_year <= 2016")["attendance"])
all_train_y2 = train_df.query("match_date_year <= 2016")["attendance"] / train_df.query("match_date_year <= 2016")["capacity"]

test_X = test_df.drop(["id", "attendance"] + obj_col + del_col, axis=1)

# smpsb_df = pd.read_csv("../input/sample_submit.csv", header=None)
# smpsb_df.iloc[:len(test_X), 1] = 0

gbms = []
for i in tqdm(range(50)):
    gbm = lgb.LGBMRegressor(n_estimators=50,
                            n_jobs=-1,
                            num_leaves=int(2**6.18),
                            feature_fraction=.63,
                            lambda_l1=10**-2.68,
                            lambda_l2=10**-2.67,
                            min_data_in_leaf=3,
                            learning_rate=10**-1.46,
                            num_boost_round=1000,
                            random_state=2434 + i)

    gbm.fit(all_train_X, all_train_y2,
            sample_weight=np.power(np.log1p(all_train_X.loc[:, "capacity"]), 2))
    # test_pred = gbm.predict(test_X) * test_X["capacity"].values
    # smpsb_df.iloc[:len(test_X), 1] += test_pred / 50
    gbms.append(gbm)

with open("models.pickle", mode="wb") as f:
    load_gbm = pickle.dump(gbms, f)
# smpsb_df.to_csv("../final_output/newstadium_oldcount.csv", index=None, header=None)