# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:47:21 2018

@author: nadare
"""

import re
import os
from time import sleep
from multiprocessing import Pool

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm


save_dir = "../ext_source/"

if not os.path.isdir(save_dir + "/html/SFMS02/"):
    os.makedirs(save_dir + "/html/SFMS02/")

sfms02_file_names = set(os.listdir(save_dir + "/html/SFMS02/"))

# SFMS01
def scraper_1(year, division):
    url = "https://data.j-league.or.jp/SFMS01/search"
    params = {"competition_years": year,
              "competition_frame_ids": division}

    r = requests.get(url, params)
    soup = BeautifulSoup(r.content, "lxml")
    result = []

    for row in soup.select("#search-list > div > table > tbody")[0]\
                           .find_all("tr"):
        data = {}
        values = row.find_all("td")
        if values[6].a is None:
            data["id"] = None
        else:
            data["id"] = values[6].a.get("href").split("=")[1]
        data["match_date"] = "{}-{}-{}".format(values[0].text,
                                               values[3].text[:2],
                                               values[3].text[3:5])
        data["division"] = division
        data["kick_off_time"] = re.sub("\s", "", values[4].text)
        data["section"] = values[2].text[:-3]
        data["round"] = values[2].text[-3:]
        data["home_team"] = re.sub("\s", "", values[5].text)
        data["away_team"] = re.sub("\s", "", values[7].text)
        data["broadcasters"] = values[10].text.replace("／", "/")
        data["attendance"] = re.sub("\s", "", values[9].text).replace(",", "")
        result.append(data)
    return result

# SFMS02
def scraper_2(id_):
    def clean(x):
        x = re.sub("(\S)\s(\S)", "\\1@\\2", x)
        x = re.sub("\s", "", x)
        return x.replace("@", " ")
    
    url = "https://data.j-league.or.jp/SFMS02/"
    params = {"match_card_id": id_}

    if "SFMS02_{}.html".format(id_) in sfms02_file_names:
        with open(save_dir + "/html/SFMS02/SFMS02_{}.html".format(id_),
                  mode="r",
                  encoding="utf-8") as f:
            html_ = f.read()
            soup = BeautifulSoup(html_, "lxml")
    else:
        r = requests.get(url, params)
        soup = BeautifulSoup(r.content, "lxml")
        with open(save_dir + "/html/SFMS02/SFMS02_{}.html".format(id_),
                  mode="w",
                  encoding="utf-8") as f:
            f.write(soup.prettify())
        sleep(1)

    values = soup.select("#contents > form > div > table > tbody > tr")[0]\
                 .find_all("td")
    data = {}
    data["id"] = id_
    data["venue"] = re.sub("\s", "", values[2].text)
    data["weather"] = re.sub("\s", "", values[4].text)
    data["temperature"] = re.sub("[^0-9.]", "", values[5].text)
    data["humidity"] = re.sub("[^0-9.]", "", values[6].text)


    selector = "#contents > form > div > div > div > div > div > table > tr"
    scores = soup.select(selector)[0]\
                 .find_all(class_="score")
    match_data = {"id":id_,
                  "home_team_score": re.sub("\D", "", scores[0].text),
                  "away_team_score": re.sub("\D", "", scores[1].text)}
    
    selector = "#contents > form > div > div > div > div > table > tbody > tr"
    members = soup.select(selector)
    for i in range(1, 12):
        member = members[i].find_all("td")
        match_data["home_team_player{}".format(i)] = "{} {} {}".format(*map(clean, [member[1].text,
                                                                                    member[2].text,
                                                                                    member[0].text]))

    for i in range(1, 12):
        member = members[12+i].find_all("td")
        match_data["away_team_player{}".format(i)] = "{} {} {}".format(*map(clean, [member[1].text,
                                                                                    member[2].text,
                                                                                    member[0].text]))
    return data, match_data


def scraper_3(stadium):
    url = "https://www.google.co.jp/search"
    params = {"q": stadium}

    r = requests.get(url, params)
    soup = BeautifulSoup(r.content, "lxml")
    data = {"stadium": stadium}

    if len(soup.select("#rhs_block")) == 0:
        return [{"stadium": stadium}]
    for div in soup.select("#rhs_block")[0].select("div > div > div"):
        spans = div.find_all("span")
        if len(spans) != 2:
            continue
        data[spans[0].text] = spans[1].text
    result = [data]
    sleep(1)
    return result


if __name__ == "__main__":
    results = []
    print("J1 load")
    for year in tqdm(range(1993, 2019)):
        results.extend(scraper_1(year, 1))
        sleep(1)
    print("J2 load")
    for year in tqdm(range(1999, 2019)):
        results.extend(scraper_1(year, 2))
        sleep(1)
    SFMS01 = pd.DataFrame(results)
    SFMS01.loc[:, "id"] = SFMS01.loc[:, "id"].astype(str)
    # SFMS01.to_csv(save_dir + "SFMS01.csv", index=None)
    # SFMS01 = pd.read_csv(save_dir + "SFMS01.csv", dtype={"id": str})

    print("match load")

    results = []
    match_report = []
    for id_ in tqdm(SFMS01.id.dropna()):
        if id_ == "None":
            continue
        data, match_data = scraper_2(id_)
        results.append(data)
        match_report.append(match_data)

    # multi core version
    # アクセス負荷につながるため、データをDLする際に使うのはやめましょう
    """
    ids = [id_ for id_ in SFMS01.id.dropna() if id_ != "None"]
    with Pool(6) as p:
       results, match_report = map(list, zip(*tqdm(p.imap(scraper_2, ids))))"""

    SFMS02 = pd.DataFrame(results)
    # SFMS02.to_csv(save_dir + "SFMS02.csv", index=None)
    ex_total = SFMS01.merge(SFMS02,
                            on="id",
                            how="left")

    ex_match_reports = pd.DataFrame(match_report)
    ex_total.to_csv(save_dir + "ex_total.csv", index=None)
    ex_match_reports.to_csv(save_dir + "ex_match_reports.csv", index=None)

    stadium_results = []
    for stadium in tqdm(ex_total["venue"].dropna().unique()):
        stadium_results.extend(scraper_3(stadium))
    ex_stadium = pd.DataFrame(stadium_results)[["stadium", "収容人数： "]]
    ex_stadium.columns = ["stadium", "capacity"]
    ex_stadium["capacity"] = ex_stadium["capacity"].fillna("").apply(lambda x: re.sub("\D", "", x))
    ex_stadium.to_csv(save_dir + "ex_stadium_capacity_mapping.csv", index=None)
