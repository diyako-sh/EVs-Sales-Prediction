# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import math
import inspect
import os
################################ Functions #################################


def automotive_crawler(query):
    """
    automotive_crawler extracts news from the Autonews website.

    Parameters
    ----------
    query : str
        make or model associated with a vehicle
    """
    driver = webdriver.Chrome(executable_path="PATH_TO_CHROMWDRIVER")
    driver.get(r"https://www.autonews.com/")
    search_box = driver.find_element_by_css_selector("#edit-search-phrase")
    search_box.clear()
    search_box.send_keys(query)
    search_box.send_keys(Keys.ENTER)
    # find news
    news = driver.find_elements_by_css_selector(
        "#mm-0 > div.has-interstitial-ads-page > div.dialog-off-canvas-main-canvas > div > main > div > div > div.layout-12coltop-2col-8x4-12colbottom.hide-block-ad.clearfix > div:nth-child(2) > div.region-featured-left.col-xs-12.col-sm-12.col-md-12.col-lg-8 > div > div.views-element-container.block.block-views.block-views-blocksearch-block > div > div > div.view-content.elastic-search-content > div > ul > li")
    # find headlines
    headlines = [i.find_element_by_css_selector(
        " div > span > div > a > h3").text for i in news]
    # find dates
    dates = [i.find_element_by_css_selector(
        "div > span > div > div.overline.clear-left-after.elastic-search-result-date > span").text for i in news]
    description = []
    # find description
    for j in news:
        try:
            description.extend([j.find_element_by_css_selector(
                " div > span > div > p:nth-child(4)").text])
        except BaseException:
            description.extend(["None"])
    des_text = driver.find_element_by_css_selector(
        "#crain-elastic-search-form > div.js-form-item.form-item.js-form-type-textfield.form-type-textfield.js-form-item-search-phrase.form-item-search-phrase > span > div.elastic-search-showing-result-wrapper").text
    # split text
    text_split1 = des_text.split("OF")
    text_split2 = text_split1[1].split("FOR")
    # find the number of news
    num = int(text_split2[0])
    # find the number of pages(25 news per page)
    num_of_pages = math.floor(num / 25)
    time.sleep(30)
    # split news
    for i in range(num_of_pages - 1):
        news = driver.find_elements_by_css_selector(
            "#mm-0 > div.has-interstitial-ads-page > div.dialog-off-canvas-main-canvas > div > main > div > div > div.layout-12coltop-2col-8x4-12colbottom.hide-block-ad.clearfix > div:nth-child(2) > div.region-featured-left.col-xs-12.col-sm-12.col-md-12.col-lg-8 > div > div.views-element-container.block.block-views.block-views-blocksearch-block > div > div > div.view-content.elastic-search-content > div > ul > li")
        new_date = pd.to_datetime(news[0].find_element_by_css_selector(
            "div > span > div > div.overline.clear-left-after.elastic-search-result-date > span").text)
        # apply a filter on news date
        if new_date > pd.to_datetime("2014-01-01"):
            next_page = driver.find_element_by_css_selector(
                "#mm-0 > div.has-interstitial-ads-page > div.dialog-off-canvas-main-canvas > div > main > div > div > div.layout-12coltop-2col-8x4-12colbottom.hide-block-ad.clearfix > div:nth-child(2) > div.region-featured-left.col-xs-12.col-sm-12.col-md-12.col-lg-8 > div > div.views-element-container.block.block-views.block-views-blocksearch-block > div > div > div.elastic-search-pager > nav > ul > li.pager__item.pager__item--next > a")
            next_page.click()
            time.sleep(5)
            news = driver.find_elements_by_css_selector(
                "#mm-0 > div.has-interstitial-ads-page > div.dialog-off-canvas-main-canvas > div > main > div > div > div.layout-12coltop-2col-8x4-12colbottom.hide-block-ad.clearfix > div:nth-child(2) > div.region-featured-left.col-xs-12.col-sm-12.col-md-12.col-lg-8 > div > div.views-element-container.block.block-views.block-views-blocksearch-block > div > div > div.view-content.elastic-search-content > div > ul > li")
            # extract headlines
            headlines.extend([i.find_element_by_css_selector(
                " div > span > div > a > h3").text for i in news])
            # extract date
            dates.extend([i.find_element_by_css_selector(
                "div > span > div > div.overline.clear-left-after.elastic-search-result-date > span").text for i in news])
            # extract description
            for j in news:
                try:
                    description.extend([j.find_element_by_css_selector(
                        " div > span > div > p:nth-child(4)").text])
                except BaseException:
                    description.extend(["None"])
        else:
            break
    # summarize extracted data in a dataframe
    news_df = pd.DataFrame({"date": dates,
                            "headline": headlines,
                            "description": description})

    news_df['date'] = pd.to_datetime(news_df.date)
    # double check for date
    for z in news_df.index:
        if news_df.date[z] < pd.to_datetime("2014-01-01"):
            number = z
            break
    news_df = news_df.iloc[0:number, :]
    # save dataframe
    news_df.to_csv(f"./news/{query}_news.csv")
    # sentiment analysis of headlines
    vader = SentimentIntensityAnalyzer()
    scores = news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    news_df = news_df.join(scores_df, rsuffix='_right')
    news_df.to_csv(f"./Analysis_by_head/{query}_news_analysis.csv")
    # sentiment analysis of description
    des_scores = news_df['description'].apply(vader.polarity_scores).tolist()
    scores_df_des = pd.DataFrame(des_scores)
    news_df = news_df.join(scores_df_des, rsuffix='_right')
    news_df.to_csv(f"./Analysis_by_des/{query}_news_analysis.csv")


################################ Run #################################
if __name__ == "__main__":
    # set workspace
    workspace = os.path.dirname(
        os.path.abspath(
            inspect.getfile(
                inspect.currentframe())))
    # create folders
    try:
        os.mkdir(workspace + "/news")
    except FileExistsError:
        pass
    try:
        os.mkdir(workspace + "/Analysis_by_head")
    except FileExistsError:
        pass
    try:
        os.mkdir(workspace + "/Analysis_by_des")

    except FileExistsError:
        pass
    # load the initial dataframe containing vehicles information(e.g. make,
    # model, segment, and category)
    make_df = pd.read_csv("./data/init_info_df.csv", index_col=0)
    for make in make_df.make:
        automotive_crawler(make)
        print(f"crawling step for {make} is done!")
