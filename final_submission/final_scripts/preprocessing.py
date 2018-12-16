# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:56:30 2018

@author: nadare
"""

# 標準ライブラリ
from calendar import isleap
from datetime import timedelta
from datetime import datetime as dt
from itertools import combinations, product, accumulate
import re
import warnings
warnings.filterwarnings("ignore")

# anacondaに入っている
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# installが必要　> pip install tqdm
from tqdm import tqdm

######################################
# フォントがある場合のみ
import matplotlib as mpl
font = {"family":"Noto Sans CJK JP"}
mpl.rc('font', **font)
######################################

def datetime_processing():
    # 日付の処理
    def day_of_year_angle(x):
        radian = 2 * np.pi * x["match_date_dayofyear"] / (365 + isleap(x["match_date_year"]))
        return radian

    def streak_length(iterable):
        streak = [0]*len(iterable)
        mode = 0
        start = 0
        for i, v in enumerate(iterable):
            if v == mode:
                continue
            elif mode == 0:
                mode = 1
                start = i
            else:
                mode = 0
                length = i - start
                for j in range(start, i):
                    streak[j] = length
        if mode == 1:
            length = len(iterable) - start
            for j in range(start, len(iterable)):
                streak[j] = length
        return streak

    ext_holiday_df = pd.read_csv("../ext_source/ext_holiday_ok.csv")
    ext_holiday_df["holiday_date"] = pd.to_datetime(ext_holiday_df["holiday_date"])

    datetimes = []
    for i in range((pd.Timestamp("2018-12-31") - pd.Timestamp("1993-01-01")).days + 1):
        datetimes.append(pd.Timestamp("1993-01-01") + timedelta(i))

    days_df = pd.DataFrame(datetimes)
    days_df.columns = ["match_date"]
    days_df["match_date_year"] = days_df["match_date"].dt.year
    days_df["match_date_month"] = days_df["match_date"].dt.month
    days_df["match_date_day"] = days_df["match_date"].dt.day
    days_df["match_date_dayofyear"] = days_df["match_date"].dt.dayofyear
    days_df["match_date_dayofweek"] = days_df["match_date"].dt.dayofweek
    days_df["match_date_sin"] = np.sin(days_df.apply(day_of_year_angle, axis=1))
    days_df["match_date_cos"] = np.cos(days_df.apply(day_of_year_angle, axis=1))

    ## 祝日の情報
    ext_holiday_df["match_date_is_holiday"] = 1
    days_df = days_df.merge(right=ext_holiday_df[["holiday_date", "match_date_is_holiday"]],
                            how="left",
                            left_on="match_date",
                            right_on="holiday_date")\
                     .drop("holiday_date", axis=1)

    ## お盆を追加
    obon_ix = days_df.query("(match_date_month == 8) & (13 <= match_date_day <= 16)").index
    days_df.loc[obon_ix, "match_date_is_holiday"] = 1
    days_df["match_date_is_holiday"].fillna(0, inplace=True)

    # 連休情報
    days_df["match_date_is_dayoff"] = ((days_df["match_date_is_holiday"] == 1)
                                   | (days_df["match_date_dayofweek"] == 5)
                                   | (days_df["match_date_dayofweek"] == 6)).astype(np.int32)
    days_df["match_date_next_day_is_dayoff"] = days_df["match_date_is_dayoff"].shift(-1).fillna(1)
    days_df["match_date_dayoff_streak"] = streak_length(days_df["match_date_is_dayoff"].values)
    
    return days_df


def total_processing(ext_total_df):
    # 全角半角の混同を解消
    def make_zen2han_dic(iterable):
        zen = list("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ" +
                   "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ" +
                   "０１２３４５６７８９，、．。（）＿−　")
        han = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,,..()_- ")
        z2h = {}
        pattern = "|".join(zen)
        for it in iterable:
            if re.search(pattern, it) is None:
                continue
            target = it*1
            for z, h in zip(zen, han):
                target = target.replace(z, h)
            z2h[it] = target
        return z2h

    for col in ["home_team", "away_team", "broadcasters", "round", "section", "venue"]:
        z2h = make_zen2han_dic(ext_total_df[col].dropna().unique().tolist())
        ext_total_df[col].replace(z2h, inplace=True)
    ext_total_df.loc[:, "section"] = ext_total_df.loc[:, "section"].apply(lambda x: int(x[1:-1]))
    ext_total_df.loc[:, "round"] = ext_total_df.loc[:, "round"].apply(lambda x: int(x[1:-1]))

    ext_total_df["kick_off_time_hour"] = ext_total_df["kick_off_time"].dt.hour
    ext_total_df["kick_off_time_minute"] = ext_total_df["kick_off_time"].dt.minute

    sec2_start = {1993: "1993-07-24",
                  1994: "1994-08-10",
                  1995: "1995-08-12",
                  1997: "1997-07-30",
                  1998: "1998-08-22",
                  1999: "1999-08-06",
                  2000: "2000-06-24",
                  2001: "2001-08-11",
                  2002: "2002-08-31",
                  2003: "2003-08-16",
                  2004: "2004-08-14",
                  2015: "2015-07-11",
                  2016: "2016-07-02"}

    for k, v in sec2_start.items():
        ix = ext_total_df.query("(division == 1)&(match_date_year == {})&(match_date >= '{}')".format(k, v)).index
        d = ext_total_df.query("(division==1)&(match_date_year == {})&(match_date < '{}')".format(k, v))["section"].max()
        ext_total_df.loc[ix, "section"] = ext_total_df.loc[ix, "section"] + d

    # チーム名を新しい方に統一
    # http://heatrock.fc2web.com/j_hensen.html
    team_rep = {"V川崎": "東京V",
                "平塚": "湘南",
                "市原": "千葉",
                "横浜M": "横浜FM",
                "F東京": "FC東京",
                "草津": "群馬"}

    ext_total_df["home_team"] = ext_total_df["home_team"].replace(team_rep)
    ext_total_df["away_team"] = ext_total_df["away_team"].replace(team_rep)

    # teamは両方まとめてベクトルに
    teams = ext_total_df["home_team"].unique().tolist()
    for team in teams:
        ext_total_df["team_" + team] = ((ext_total_df["home_team"] == team)
                                        | (ext_total_df["away_team"] == team))

    # 同じスタジアムはすべて名前を統一
    # 同じスタジアムはすべて名前を統一
    stadium_map = {'長崎県立総合運動公園陸上競技場': 'トランスコスモススタジアム長崎',
                   '鳥取市営サッカー場バードスタジアム': 'とりぎんバードスタジアム',
                   '大阪長居スタジアム': 'ヤンマースタジアム長居',
                   '大阪長居第2陸上競技場': 'キンチョウスタジアム',
                   '愛媛県総合運動公園陸上競技場': 'ニンジニアスタジアム',
                   '九州石油ドーム': '大分銀行ドーム',
                   'ひとめぼれスタジアム宮城': '宮城スタジアム',
                   '大分スポーツ公園総合競技場': '大分銀行ドーム',
                   'サンプロアルウィン': '松本平広域公園総合球技場',
                   '市立吹田サッカースタジアム': 'パナソニックスタジアム吹田',
                   'アウトソーシングスタジアム日本平': 'IAIスタジアム日本平',
                   '維新百年記念公園陸上競技場': '維新みらいふスタジアム',
                   '平塚競技場': 'ShonanBMWスタジアム平塚',
                   '等々力緑地運動公園陸上競技場': '等々力陸上競技場',
                   '名古屋市瑞穂球技場': 'パロマ瑞穂スタジアム',
                   '長居第2陸上競技場': 'キンチョウスタジアム',
                   '徳島県鳴門総合運動公園陸上競技場': '鳴門・大塚スポーツパークポカリスエットスタジアム',
                   '日本平運動公園球技場': 'IAIスタジアム日本平',
                   '群馬県立敷島公園県営陸上競技場': '正田醤油スタジアム群馬',
                   '埼玉県営大宮公園サッカー場': 'NACK5スタジアム大宮',
                   '横浜市三ツ沢総合公園球技場': 'ニッパツ三ツ沢球技場',
                   '札幌厚別運動公園競技場': '札幌厚別公園競技場',
                   'ジュビロ磐田スタジアム': 'ヤマハスタジアム(磐田)',
                   '浦和市駒場スタジアム': '浦和駒場スタジアム',
                   '博多の森陸上競技場': 'レベルファイブスタジアム',
                   'ジュビロ磐田サッカースタジアム': 'ヤマハスタジアム(磐田)',
                   '西が丘サッカー場': '味の素フィールド西が丘',
                   '国立霞ヶ丘競技場': '国立競技場',
                   '熊本県民総合運動公園陸上競技場': 'えがお健康スタジアム',
                   'kankoスタジアム': 'シティライトスタジアム',
                   '横浜市三ツ沢公園球技場': 'ニッパツ三ツ沢球技場',
                   '山形県総合運動公園陸上競技場': 'NDソフトスタジアム山形',
                   '鳥栖スタジアム': 'ベストアメニティスタジアム',
                   '京都市西京極総合運動公園陸上競技場': '京都市西京極総合運動公園陸上競技場兼球技場',
                   '長野市営長野運動公園総合運動場': '長野運動公園総合運動場',
                   '岡山県陸上競技場桃太郎スタジアム': 'シティライトスタジアム',
                   '静岡県営草薙陸上競技場': '草薙総合運動公園陸上競技場',
                   'うまかな・よかなスタジアム': 'えがお健康スタジアム',
                   '瑞穂公園陸上競技場': 'パロマ瑞穂スタジアム',
                   '国立西が丘サッカー場': '味の素フィールド西が丘',
                   '東京スタジアム': '味の素スタジアム',
                   '白波スタジアム': '鹿児島県立鴨池陸上競技場',
                   '新潟スタジアム': 'デンカビッグスワンスタジアム',
                   '千代台公園陸上競技場': '函館市千代台公園陸上競技場',
                   '日本平スタジアム': 'IAIスタジアム日本平',
                   '水戸市立競技場': 'ケーズデンキスタジアム水戸',
                   'ホームズスタジアム神戸': 'ノエビアスタジアム神戸',
                   '日立柏サッカー場': '三協フロンテア柏スタジアム',
                   '横浜国際総合競技場': '日産スタジアム',
                   '静岡スタジアムエコパ': 'エコパスタジアム',
                   '長居スタジアム': 'ヤンマースタジアム長居',
                   '東平尾公園博多の森球技場': 'レベルファイブスタジアム',
                   'さいたま市浦和駒場スタジアム': '浦和駒場スタジアム',
                   '駒沢陸上競技場': '駒沢オリンピック公園総合運動場陸上競技場',
                   '東北電力ビッグスワンスタジアム': 'デンカビッグスワンスタジアム',
                   'さいたま市大宮公園サッカー場': 'NACK5スタジアム大宮',
                   '山梨県小瀬スポーツ公園陸上競技場': '山梨中銀スタジアム',
                   '七北田公園仙台スタジアム': 'ユアテックスタジアム仙台',
                   '国立スポーツ科学センター西が丘サッカー場': '味の素フィールド西が丘',
                   '静岡県草薙総合運動場陸上競技場': '草薙総合運動公園陸上競技場',
                   '仙台スタジアム': 'ユアテックスタジアム仙台',
                   '広島ビッグアーチ': 'エディオンスタジアム広島',
                   '東平尾公園博多の森陸上競技場': 'レベルファイブスタジアム',
                   '広島スタジアム': 'コカ・コーラウエスト広島スタジアム',
                   '神戸ウイングスタジアム': 'ノエビアスタジアム神戸',
                   '名古屋市瑞穂陸上競技場': 'パロマ瑞穂スタジアム',
                   '香川県立丸亀競技場': 'Pikaraスタジアム'}

    ext_total_df.loc[:, "venue"] = ext_total_df["venue"].replace(stadium_map)

    # weather
    # シンプルに単語を含んでいるか否かで判定(BoWみたい)
    ext_total_df["weather"] = ext_total_df["weather"].fillna("")
    for condition in ["晴", "曇", "雨", "屋内", "のち", "時々", "一時"]:
        ext_total_df["weather_{}".format(condition)] = ext_total_df["weather"].str.contains(condition)
    ext_total_df["weather_その他"] = ext_total_df["weather"].str.contains("雪|雷|霧")

    # 温度と気温から不快指数を計算
    T = ext_total_df["temperature"]
    H = ext_total_df["humidity"]
    ext_total_df["THI"] = 0.81*T + 0.01*H*(0.99*T - 14.3) + 46.3

    # broad_casters
    # ()と※より後ろを消す
    # 小文字を大文字に統一
    # ーを-に変換する
    # スカパー派生は全てスカパーに統一
    # 録画か否かは無視
    def cleanify(caster):
        caster = caster.upper()
        caster = re.sub("\(.+\)|\*", "", caster)
        caster = caster.split("※")[0]\
                       .rstrip(" ")\
                       .strip()\
                       .replace("ー", "-")\
                       .replace("－", "-")\
                       .replace("＊", "")\
                       .replace("放送局", "")\
                       .replace(" ", "_")\
                       .split("18")[0]
        if caster == 'E2録':
            caster = "E2"
        return caster

    def tv_groupify(caster):
        def isbs(caster):
            if caster == "BS":
                return True
            elif caster[:3] in ["BS-", "BS2"]:
                return True
            elif caster[:6] in ["NHK-BS", "NHK_BS"]:
                return True
            else:
                return False

        def iscs(caster):
            if caster[0] in list("DEJスデ"):
                return True
            else:
                return False

        caster = caster.replace("[録]", "")

        if caster == "":
            return -1
        if caster in ["NHK総合", "TBS", "テレビ朝日", "フジテレビ", "日本テレビ"]:
            group = 0
        elif isbs(caster):
            group = 1
        elif iscs(caster):
            group = 2
        else:
            group = 3

        return group  # *2 + recorded 録画系は2016年以降少なかったので無視

    dirty_casters = set()
    for casters in ext_total_df["broadcasters"].dropna()\
                                               .str.replace(",BS", "/BS")\
                                               .str.replace(",静岡", "/静岡")\
                                               .str.split("/"):
        dirty_casters.update(casters)

    clean_casters = set()
    for caster in dirty_casters:
        clean_caster = cleanify(caster)
        clean_casters.add((clean_caster, tv_groupify(clean_caster)))

    ext_total_df["broadcasters"] = ext_total_df["broadcasters"].fillna("")
    tmp = ext_total_df.loc[:, "broadcasters"].str.replace(",BS","/BS")\
                                             .str.replace(",静岡", "/静岡")\
                                             .str.split("/")\
                                             .apply(lambda casters: [cleanify(caster) for caster in casters])
    groups = tmp.apply(lambda clean_casters: [tv_groupify(caster) for caster in clean_casters])
    ext_total_df.loc[:, "broadcastersgroup"] = groups
    
    for i in range(4):
        ext_total_df.loc[:, "broadcastersgroup_{}".format(i)] = ext_total_df.loc[:, "broadcastersgroup"]\
                                                                            .apply(lambda group: group.count(i))

    ext_total_df["broadcasters_size"] = ext_total_df.filter(regex="^broadcastersgroup_").sum(axis=1)
    ext_total_df["broadcasters_anysize"] = (ext_total_df.filter(regex="^broadcastersgroup_")>0).sum(axis=1)

    return ext_total_df


def distance_processing():
    def hubeny(lng1, lat1, lng2, lat2):
        # http://www.trail-note.net/tech/calc_distance/
        # 座標から距離を計算する

        # WGS84
        Rx = 6378137.000
        Ry = 6356752.314140

        Dx = (lat1 - lat2)/360*2*np.pi
        Dy = (lng1 - lng2)/360*2*np.pi

        P = (lng1 + lng2)/360*np.pi

        E = np.sqrt((np.power(Rx, 2) - np.power(Ry, 2))/np.power(Rx, 2))
        W = np.sqrt(1-np.power(E, 2)*np.power(np.sin(P), 2))
        M = (Rx*(1-np.power(E, 2)))/(np.power(W, 3))
        N = Rx/W

        D = np.sqrt(np.power(Dy*M, 2) + np.power(Dx*N*np.cos(P), 2))
        return D

    team_loc_df = pd.read_csv("../ext_source/team_where.csv")
    team_loc_df["lng"] = team_loc_df["LL"].apply(lambda x: float(x.split(",")[0]))
    team_loc_df["lat"] = team_loc_df["LL"].apply(lambda x: float(x.split(",")[1]))

    results = []
    for A, B in combinations(team_loc_df[["team", "lng", "lat"]].values, 2):
        dist = hubeny(A[1], A[2], B[1], B[2])
        results.append({"home_team": A[0],
                        "away_team": B[0],
                        "distance": dist})
        results.append({"home_team": B[0],
                        "away_team": A[0],
                        "distance": dist})

    distance_pair_df = pd.DataFrame(results)

    return distance_pair_df


def match_processing(ext_total_df):

    def simulate_matching(match_df):
        # 2003年以降のJ1方式で計算する
        # 試合開始前の情報で考える
        # 19188 : G大阪 vs 鹿島 2017-13節の試合を17節の後に実施
        # 19187 : 川崎F vs 浦和 2017-13節の試合を17節の後に実施
        # 19264: C大阪 vs 浦和　2017-22節の試合を18節の後に実施*問題なし
        # 20870: C大阪 vs 鹿島 2018-14節の試合を17節の後に実施
        # 20899: 横浜FM vs 清水 2018-18節の試合を24節の後に実施
        #      : 湘南 vs 川崎F 2018-18節の試合を27節の後に実施
        #      : 名古屋 vs 札幌 2018-18節の試合を30節の後に実施
        #      : 磐田 vs 湘南 2018-28節の試合を30節の後に実施
        #      : C大阪 vs 名古屋 2018-28節の試合を31節の後に実施

        results = []
        for year, division in product(sorted(match_df["match_date_year"].unique()), [1, 2]):
            year_df = match_df.loc[(match_df["match_date_year"]==year) & (match_df["division"] == division)]
            teams = year_df.home_team.unique().tolist()
            sec_size = year_df.section.max()
            points = {team:[[0]*(sec_size+1), [0]*(sec_size+1), [0]*(sec_size+1)] for team in teams}
            columns = ["home_team", "away_team", "home_team_score", "away_team_score", "section"]
            for values in year_df[columns].sort_values("section").values:
                home_team, away_team, home_team_score, away_team_score, section = values
                home_team_score, away_team_score, section = map(int, [home_team_score, away_team_score, section])
                if home_team_score > away_team_score:
                    points[home_team][0][section] = 3
                    points[away_team][0][section] = 0
                elif home_team_score == away_team_score:
                    points[home_team][0][section] = 1
                    points[away_team][0][section] = 1
                else:
                    assert home_team_score < away_team_score
                    points[home_team][0][section] = 0
                    points[away_team][0][section] = 3

                points[home_team][1][section] = home_team_score - away_team_score
                points[away_team][1][section] = away_team_score - home_team_score

                points[home_team][2][section] = home_team_score
                points[away_team][2][section] = away_team_score

            # 開催日時がめんどくさいやつはとりあえず0vs0で埋め、後日追加する
            if year == 2017 and division == 1:
                for home_team, away_team, true_sec, move_sec in (["G大阪", "鹿島", 13, 17],
                                                                 ["川崎F", "浦和", 13, 17]):
                    points[home_team][0][move_sec] += points[home_team][0][true_sec] - 1
                    points[away_team][0][move_sec] += points[away_team][0][true_sec] - 1
                    points[home_team][1][move_sec] += points[home_team][1][true_sec]
                    points[away_team][1][move_sec] += points[away_team][1][true_sec]
                    points[home_team][2][move_sec] += points[home_team][2][true_sec]
                    points[away_team][2][move_sec] += points[away_team][2][true_sec]

                    points[home_team][0][true_sec] = 1
                    points[away_team][0][true_sec] = 1
                    points[home_team][1][true_sec] = 0
                    points[away_team][1][true_sec] = 0
                    points[home_team][2][true_sec] = 0
                    points[away_team][2][true_sec] = 0

            elif year == 2018 and division == 1:
                for home_team, away_team, true_sec, move_sec in (["C大阪", "鹿島", 14, 17],):
                                                                 #["横浜FM", "清水", 18, 24],
                                                                 #["湘南", "川崎F", 18, 27]):
                                                                 # ["名古屋", "札幌", 18, 30],
                                                                 # ["磐田", "湘南", 28, 30],
                                                                 # ["C大阪", "名古屋", 28, 31]):
                    points[home_team][0][move_sec] += points[home_team][0][true_sec] - 1
                    points[away_team][0][move_sec] += points[away_team][0][true_sec] - 1
                    points[home_team][1][move_sec] += points[home_team][1][true_sec]
                    points[away_team][1][move_sec] += points[away_team][1][true_sec]
                    points[home_team][2][move_sec] += points[home_team][2][true_sec]
                    points[away_team][2][move_sec] += points[away_team][2][true_sec]
                    
                    points[home_team][0][true_sec] = 1
                    points[away_team][0][true_sec] = 1
                    points[home_team][1][true_sec] = 0
                    points[away_team][1][true_sec] = 0
                    points[home_team][2][true_sec] = 0
                    points[away_team][2][true_sec] = 0

            for team, p in points.items():
                p = list(map(list, map(accumulate, p)))
                for i in range(len(p[0])):
                    results.append({"section": i+1, # ひとつ前の試合の結果を繋げる
                                    "team": team,
                                    "division": division,
                                    "match_point_prevmatch": points[team][0][i],
                                    "match_point_cum": p[0][i],
                                    "match_point_delta_cum": p[1][i],
                                    "match_point_score_cum": p[2][i],
                                    "match_date_year": year})
        return results


    # 試合結果から勝ち点を再現
    ext_score_df = pd.read_csv("../ext_source/ex_match_reports.csv")[["id", "home_team_score", "away_team_score"]]
    ext_score_df.loc[:, "id"] = ext_score_df.loc[:, "id"].astype(str)
    dammy_match = [{'id': '30000', 'home_team_score': 3, 'away_team_score': 3},
                   {'id': '30001', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30002', 'home_team_score': 1, 'away_team_score': 2},
                   {'id': '30003', 'home_team_score': 2, 'away_team_score': 2},
                   {'id': '30004', 'home_team_score': 1, 'away_team_score': 2},
                   {'id': '30005', 'home_team_score': 3, 'away_team_score': 1},
                   {'id': '30006', 'home_team_score': 3, 'away_team_score': 1},
                   {'id': '30007', 'home_team_score': 2, 'away_team_score': 1},
                   {'id': '30008', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30009', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30010', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30011', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30012', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30013', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30014', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30015', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30016', 'home_team_score': 1, 'away_team_score': 1},
                   {'id': '30017', 'home_team_score': 1, 'away_team_score': 1}]
    ext_score_df = pd.concat([ext_score_df,
                              pd.DataFrame(dammy_match, columns=["id",
                                                                 "home_team_score",
                                                                 "away_team_score"])])\
                     .reset_index(drop=True)

    match_df = ext_score_df.merge(ext_total_df[["id", "home_team", "away_team",
                                                "match_date_year", "section",
                                                "round", "division"]],
                                  on="id",
                                  how="left")
    res = simulate_matching(match_df)
    match_sim_df = pd.DataFrame(res)

    match_point_columns = ["match_point_cum",
                           "match_point_delta_cum",
                           "match_point_score_cum"]
    tmp = match_sim_df.groupby(["match_date_year", "division", "section"])\
                      .apply(lambda x: x.sort_values(match_point_columns, ascending=False))

    # 順位の計算
    tmp["rank"] = tmp.groupby(level=[0, 1, 2])["match_date_year"]\
                     .rank(method="first")
    match_sim_df = tmp.reset_index(drop=True)
    match_sim_df.loc[match_sim_df.section == 1, "rank"] = -1
    match_sim_df.loc[match_sim_df.query("division == 2").index, "rank"] += 18

    # 直近n試合の戦績
    # 勝ちが続いている時はスタジアムで応援したくなるし、負けこんでいれば来なくなりそう
    piv = tmp.reset_index(drop=True)\
             .pivot_table(columns=["match_date_year", "team"],
                          index="section",
                          values="match_point_prevmatch",
                          aggfunc="sum")
    piv.iloc[0, :] = np.NaN
    recent_3_match = piv.rolling(window=3, min_periods=2)\
                        .mean()\
                        .unstack()\
                        .dropna()\
                        .reset_index()
    recent_5_match = piv.rolling(window=5)\
                        .mean()\
                        .unstack()\
                        .dropna()\
                        .reset_index()
    recent_3_match.columns = ['match_date_year',
                              'team',
                              'section',
                              "match_point_prev3_mean"]
    recent_5_match.columns = ['match_date_year',
                              'team',
                              'section',
                              "match_point_prev5_mean"]

    match_sim_df = match_sim_df.merge(recent_3_match,
                                      on=['match_date_year', 'team', 'section'],
                                      how="left")
    match_sim_df = match_sim_df.merge(recent_5_match,
                                      on=['match_date_year', 'team', 'section'],
                                      how="left")

    # 欠損値は-1でとりあえず補完
    new_col = ["match_point_prev3_mean", "match_point_prev5_mean"]
    match_sim_df.loc[:, new_col] = match_sim_df.loc[:, new_col].fillna(-1)

    # 前年度の結果
    year_final_results = []
    for year, div, section in match_sim_df.groupby(["match_date_year", "division"])["section"].max().reset_index().values:
        query = "(match_date_year == {}) & (division == {}) & (section == {})".format(year, div, section)
        for team in match_sim_df.query(query)["team"].unique():
            rank = match_sim_df.query(query + "& (team == '{}')".format(team))["rank"].values[0]
            year_final_results.append({"match_date_year": year + 1,
                                       "team": team,
                                       "rank": rank})
    prev_year_rank_df = pd.DataFrame(year_final_results).rename(columns={"rank": "prev_year_rank"})
    match_sim_df = match_sim_df.merge(prev_year_rank_df,
                                      how="left",
                                      on=["match_date_year", "team"])

    # Jリーグ初参加のチームorJ3から上がってきたチームは最下位からのスタート
    fill_dic = dict(prev_year_rank_df.groupby("match_date_year")["prev_year_rank"].max().items())
    fill_dic[1993] = -1
    ix = match_sim_df["prev_year_rank"].isna()
    match_sim_df.loc[ix, "prev_year_rank"] = match_sim_df.loc[ix, "match_date_year"].map(fill_dic)
    match_df = match_df.merge(match_sim_df,
                              how="left",
                              left_on=["match_date_year", "section", "home_team", "division"],
                              right_on=["match_date_year", "section", "team", "division"])\
                       .drop(["team"], axis=1)\
                       .rename(columns={"match_point_cum": "home_team_match_point_cum",
                                        "match_point_delta_cum": "home_team_match_point_delta_cum",
                                        "match_point_score_cum": "home_team_match_point_score_cum",
                                        'match_point_prevmatch': "home_team_match_point_prevmatch",
                                        'match_point_prev3_mean': "home_team_match_point_prev3_mean",
                                        'match_point_prev5_mean': "home_team_match_point_prev5_mean",
                                        "prev_year_rank": "home_team_prev_year_rank",
                                        "rank": "home_team_rank"})

    match_df = match_df.merge(match_sim_df,
                              how="left",
                              left_on=["match_date_year", "section", "away_team", "division"],
                              right_on=["match_date_year", "section", "team", "division"],
                              suffixes=["", "home_team_"])\
                       .drop(["team"], axis=1)\
                       .rename(columns={"match_point_cum": "away_team_match_point_cum",
                                        "match_point_delta_cum": "away_team_match_point_delta_cum",
                                        "match_point_score_cum": "away_team_match_point_score_cum",
                                        'match_point_prevmatch': "away_team_match_point_prevmatch",
                                        'match_point_prev3_mean': "away_team_match_point_prev3_mean",
                                        'match_point_prev5_mean': "away_team_match_point_prev5_mean",
                                        "prev_year_rank": "away_team_prev_year_rank",
                                        "rank": "away_team_rank"})

    match_df["rank_dif"] = match_df["home_team_rank"] - match_df["away_team_rank"]
    match_df["rank_abs"] = np.abs(match_df["home_team_rank"] - match_df["away_team_rank"])
    match_df["rank_mean"] = (match_df["home_team_rank"] + match_df["away_team_rank"])/2

    match_df["prev_year_rank_dif"] = match_df["home_team_prev_year_rank"] - match_df["away_team_prev_year_rank"]
    match_df["prev_year_rank_abs"] = np.abs(match_df["home_team_prev_year_rank"] - match_df["away_team_prev_year_rank"])
    match_df["prev_year_rank_mean"] = (match_df["home_team_prev_year_rank"] + match_df["away_team_prev_year_rank"])/2

    match_df["match_point_cum_dif"] = match_df["home_team_match_point_cum"] - match_df["away_team_match_point_cum"]
    match_df["match_point_cum_abs"] = np.abs(match_df["match_point_cum_dif"])

    match_df = match_df[['id',
                         'home_team_match_point_prevmatch',
                         'home_team_match_point_prev3_mean',
                         'home_team_match_point_prev5_mean',
                         'home_team_rank',
                         'home_team_prev_year_rank',
                         'away_team_match_point_prevmatch',
                         'away_team_match_point_prev3_mean',
                         'away_team_match_point_prev5_mean',
                         'away_team_rank',
                         'away_team_prev_year_rank',
                         'rank_dif',
                         'rank_abs',
                         'rank_mean',
                         'prev_year_rank_dif',
                         'prev_year_rank_abs',
                         'prev_year_rank_mean',
                         'match_point_cum_dif',
                         'match_point_cum_abs']]

    return match_df


def kaiji_processing(ext_total_df):
    def kessan_year(team, date):
        if date.year <= 2017:
            if team in ["柏", "磐田"]:
                res = date.year - 1 - (date < pd.Timestamp(date.year, 7, 21))
            else:
                res = date.year - 1 - (date < pd.Timestamp(date.year, 5, 27))
        else:
            if team in ["柏", "磐田"]:
                res = date.year - 1 - (date < pd.Timestamp(date.year, 7, 30))
            else:
                res = date.year - 1 - (date < pd.Timestamp(date.year, 5, 26))
        return res

    team_rep = {"V川崎": "東京V",
                "平塚": "湘南",
                "市原": "千葉",
                "横浜M": "横浜FM",
                "F東京": "FC東京",
                "草津": "群馬"}
    kaiji = pd.read_csv("../ext_source/kaiji.csv").iloc[:, :6]
    kaiji.loc[:, "team"] = kaiji["team"].replace(team_rep)

    tmp = ext_total_df[["home_team", "away_team", "match_date"]]
    tmp.loc[:, "home_team_kaiji_year"] = tmp.apply(lambda x: kessan_year(x["home_team"], x["match_date"]), axis=1)
    tmp.loc[:, "away_team_kaiji_year"] = tmp.apply(lambda x: kessan_year(x["away_team"], x["match_date"]), axis=1)

    for col in ["labor", "advertising_income", "admission_income"]:
        kaiji.loc[:, col] = kaiji[col].apply(lambda x: int(re.sub("\D", "", x)))

    tmp = tmp.merge(kaiji[["team", "year", "labor", "advertising_income", "admission_income"]],
                    left_on=["home_team", "home_team_kaiji_year"],
                    right_on = ["team", "year"],
                    how="left")\
             .rename(columns={"labor": "home_team_labor",
                              "advertising_income": "home_team_advertising_income",
                              "admission_income": "home_team_admission_income"})\
             .drop(["team", "year"], axis=1)

    tmp = tmp.merge(kaiji[["team", "year", "labor", "advertising_income", "admission_income"]],
                    left_on=["away_team", "away_team_kaiji_year"],
                    right_on = ["team", "year"],
                    how="left")\
             .rename(columns={"labor": "away_team_labor",
                              "advertising_income": "away_team_advertising_income",
                              "admission_income": "away_team_admission_income"})\
             .drop(["team", "year"], axis=1)
    return tmp


def mean_encoding(ext_total_df):
    ext_total_df["attendance_ratio"] = ext_total_df["attendance"] / ext_total_df["capacity"]

    for window in [3, 5]:
        piv_count = ext_total_df.pivot_table(columns="venue",
                                             index="match_date_year",
                                             values="attendance_ratio",
                                             aggfunc="count").fillna(1e-9)
        piv_sum = ext_total_df.pivot_table(columns="venue",
                                           index="match_date_year",
                                           values="attendance_ratio",
                                           aggfunc="sum").fillna(0)
        piv_count.loc[2017:2019, :] = 1e-09
        piv_sum.loc[2017:2019, :] = 0
        piv_rcount = piv_count.rolling(window=window, center=False)\
                              .sum()\
                              .shift(1)\
                              .fillna(0)\
                              .unstack()\
                              .reset_index()
        piv_rmean = (piv_sum.rolling(window=window, center=False).sum()
                     / piv_count.rolling(window=window, center=False).sum())\
                        .shift(1)\
                        .fillna(0)\
                        .unstack()\
                        .reset_index()

        piv_rmean.columns = ["venue", "match_date_year", "venue_rolling_mean_" + str(window)]
        piv_rcount.columns = ["venue", "match_date_year", "venue_rolling_count_" + str(window)]
        ext_total_df = ext_total_df.merge(piv_rmean,
                                          on=["venue", "match_date_year"],
                                          how="left")
        ext_total_df = ext_total_df.merge(piv_rcount,
                                          on=["venue", "match_date_year"],
                                          how="left")

    # 過去五年のある組み合わせでの成績
    window = 5
    piv_count = ext_total_df.pivot_table(columns=["home_team", "away_team"],
                                        index="match_date_year",
                                        values="attendance_ratio",
                                        aggfunc="count").fillna(1e-9)
    piv_sum = ext_total_df.pivot_table(columns=["home_team", "away_team"],
                                       index="match_date_year",
                                       values="attendance_ratio",
                                       aggfunc="sum").fillna(0)
    piv_count.loc[2017:2019, : ] = 1e-09
    piv_sum.loc[2017:2019, : ] = 0

    piv_rcount = piv_count.rolling(window=window, center=False).sum().shift(1).fillna(0).unstack().reset_index()
    piv_rmean = (piv_sum.rolling(window=window, center=False).sum() / piv_count.rolling(window=window, center=False).sum()).shift(1).fillna(0).unstack().reset_index()

    piv_rmean.columns = ["home_team", "away_team", "match_date_year", "team_pair_rolling_mean_" + str(window)]
    piv_rcount.columns = ["home_team", "away_team", "match_date_year", "team_pair_rolling_count_" + str(window)]

    ext_total_df = ext_total_df.merge(piv_rmean,
                                      on=["home_team", "away_team", "match_date_year"],
                                      how="left")
    ext_total_df = ext_total_df.merge(piv_rcount,
                                      on=["home_team", "away_team", "match_date_year"],
                                      how="left")

    ext_total_df.drop("attendance_ratio", axis=1, inplace=True)
    return ext_total_df


def stacking(ext_total_df):
    cat_col = [15, 18]
    lin_col = [3, 9, 10, 11, 14, 17, 21, 22, 23, 24, 25] + list(range(27, 81)) + list(range(82, 108)) # ここまで欠損値無し？
    nan_is_m1 = ["home_team_match_point_prev3_mean",
                 "home_team_match_point_prev5_mean",
                 "away_team_match_point_prev3_mean",
                 "away_team_match_point_prev5_mean"]

    lin_total_df = ext_total_df.iloc[:, [0] + lin_col]
    lin_total_df = pd.concat([lin_total_df,
                              pd.get_dummies(ext_total_df.iloc[:, cat_col].astype("str"), drop_first=True)], axis=1)
    lin_total_df.loc[:, nan_is_m1] = lin_total_df.loc[:, nan_is_m1].replace(-1, 1.2)

    dropcol = ["attendance"]
    all_train_X = lin_total_df.query("1994 <= match_date_year").drop(dropcol, axis=1)
    all_train_y = np.log1p(lin_total_df.query("1994 <= match_date_year")["attendance"])
    all_train_y2 = lin_total_df.query("1994 <= match_date_year")["attendance"] / lin_total_df.query("1994 <= match_date_year")["capacity"]

    for year in tqdm(range(2002, 2017, 2)):
        window = 6
        duration = 2

        Z = all_train_X.match_date_year
        Z2 = all_train_X.division

        train = (year-duration-window<Z)&(Z<=year-duration)
        val = (year - duration < Z) & (Z <= year)

        train_X = all_train_X.loc[train, :]
        train_y = all_train_y[train]
        train_y2 = all_train_y2[train]

        val_X = all_train_X.loc[val, :]

        scl = StandardScaler()
        scl.fit(train_X.values.astype(np.float64))
        train_scl = scl.transform(train_X.values.astype(np.float64))
        val_scl = scl.transform(val_X.values.astype(np.float64))

        elastic_net = ElasticNet(alpha=10**-2.7,
                                 l1_ratio=0.75,
                                 max_iter=10000,
                                 random_state=2434)

        elastic_net.fit(train_scl, train_y2)
        ext_total_df.loc[val_X.index, "elastic_net"] = elastic_net.predict(val_scl).clip(0, 1)

    train = (2010<Z) & (Z<=2016)
    val = (2017==Z) |  ((Z == 2018) & ((all_train_X["section"] <= 17)|(33<=all_train_X["section"])))

    train_X = all_train_X.loc[train, :]
    train_y = all_train_y[train]
    train_y2 = all_train_y2[train]

    val_X = all_train_X.loc[val, :]

    scl = StandardScaler()
    scl.fit(train_X.values.astype(np.float64))
    train_scl = scl.transform(train_X.values.astype(np.float64))
    val_scl = scl.transform(val_X.values.astype(np.float64))

    elastic_net = ElasticNet(alpha=10**-2.7,
                             l1_ratio=0.75,
                             random_state=2434)

    elastic_net.fit(train_scl, train_y2)
    ext_total_df.loc[val_X.index, "elastic_net"] = elastic_net.predict(val_scl).clip(0, 1)
    return ext_total_df


if __name__ == "__main__":
    ext_total_df = pd.read_csv("../ext_source/ex_total.csv")
    test_df = pd.read_csv("../input/test.csv")
    dammy_weathers = [[8.2, '曇時々晴', 62],
                      [12.9, '曇時々晴', 49],
                      [14.0, '晴時々曇', 48],
                      [16.0, '曇時々晴', 44],
                      [16.0, '曇時々晴', 44],
                      [14.1, '曇時々晴', 52],
                      [14.1, '曇時々晴', 52],
                      [13.7, '曇時々晴', 46],
                      [15.0, '曇時々晴', 28],
                      [21.1, '屋内', 42.7],
                      [16.5, '晴', 60],
                      [17.6, '晴', 50],
                      [17.8, '晴', 56],
                      [17.9, '晴', 42],
                      [17.9, '晴', 42],
                      [18.6, '晴', 58],
                      [21.06, '屋内', 66],
                      [19.8, '晴', 52]]

    ext_total_df.loc[7332:7349, "id"] = list(map(str, range(30000, 30018)))
    ext_total_df.loc[7332:7349, "venue"] = test_df.tail(18)["venue"].values
    ext_total_df.loc[7332:7349, ["temperature", "weather", "humidity"]] = dammy_weathers

    ext_total_df.rename(columns={"broad_casters": "broadcasters"}, inplace=True)
    ext_total_df.loc[:, "match_date"] = pd.to_datetime(ext_total_df["match_date"])
    ext_total_df.loc[:, "kick_off_time"] = pd.to_datetime(ext_total_df["kick_off_time"])

    print("datetime processing: ", end="")
    tic = dt.now()
    days_df = datetime_processing()
    ext_total_df = ext_total_df.merge(days_df,
                                      how="left",
                                      on="match_date")
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))
    # days_df.to_csv("../processed/days_df.csv", index=None)

    print("total_df processing: ", end="")
    tic = dt.now()
    ext_total_df = total_processing(ext_total_df)
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))

    print("distance processing: ", end="")
    tic = dt.now()
    distance_pair_df = distance_processing()
    ext_total_df = ext_total_df.merge(distance_pair_df,
                                      on=["home_team", "away_team"],
                                      how="left")
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))
    # distance_pair_df.to_csv("../processed/distance_pair_df.csv", index=None)

    print("stadium processing: ", end="")
    tic = dt.now()
    ext_stadium_df = pd.read_csv("../ext_source/ex_modified_stadium2.csv")[["stadium", "capacity"]]
    ext_total_df = ext_total_df.merge(right=ext_stadium_df,
                                      how="left",
                                      left_on="venue",
                                      right_on="stadium")\
                               .drop("stadium", axis=1)
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))

    print("match processing: ", end="")
    tic = dt.now()
    match_df = match_processing(ext_total_df)
    ext_total_df = ext_total_df.merge(match_df,
                                      on="id",
                                      how="left")
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))
    # match_df.to_csv("../processed/match_df.csv", index=None)

    print("決算情報 processing: ", end="")
    tic = dt.now()
    kaiji_total_df = kaiji_processing(ext_total_df)
    ext_total_df = pd.concat([ext_total_df, kaiji_total_df.iloc[:, 5:].fillna(-1)], axis=1)
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))
    # kaiji_total_df.to_csv("../processed/kaiji_total_df.csv", index=None)

    print("mean_encoding/stacking: ")
    tic = dt.now()
    ext_total_df.drop(ext_total_df.query("attendance <= 100").index, inplace=True)
    ext_total_df = mean_encoding(ext_total_df)
    ext_total_df = stacking(ext_total_df)
    toc = dt.now()
    print("{}ms".format((toc-tic).seconds*1000 + (toc-tic).microseconds // 1000))

    ext_total_df.to_hdf("../final_processed/newstadium_oldcount.h5", "data")