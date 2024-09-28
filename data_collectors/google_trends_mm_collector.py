# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import inspect
import os
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.colors as mcolors
################################ Functions #################################


def trends_line_plot(
        df,
        x_name,
        color_dict,
        xlabel="date",
        ylabel="shopper",
        file_name=None,
        path=None,
        figureSize=(
            12,
            8),
        xticks_number=5):
    """
    trends_line_plot generates line plot for trends.

    Parameters
    ----------
    df : dataframe
        input dataframe containing trend data
    x_name : str
        the name of x data column
    color_dict : dict
        a dictionary containing color names for each trend
    xlabel : str
        xlabel, by default "date"
    ylabel : str
        ylabel, by default "shopper"
    file_name : str
        the file name for saving
    path : str
        the path for saving
    figureSize : tuple
        figureSize, by default (12,8)
    xticks_number : int
        the number of xticks, by default 5
    """
    fig, ax = plt.subplots(figsize=figureSize)
    for y_name in list(color_dict.keys()):
        color = color_dict[y_name]
        ax.plot(df[x_name], df[y_name], alpha=0.8, linewidth=.8, color=color)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.tick_params("x", labelsize=7, labelrotation=45)
    myFmt = mdates.DateFormatter('%d %b %Y')
    ax.xaxis.set_major_formatter(myFmt)
    legned = list(color_dict.keys())
    ax.legend(legned)
    M = len(df) - 1
    cl = int(M / xticks_number)
    ind = list(range(M + 1)[int((M % cl) / 2)::cl])
    ind[1], ind[-1] = M, 0
    del cl, M
    ax.set_xticks(list(df.iloc[ind][x_name].values))
    fig.savefig(path + "/" + file_name, dpi=600)


def selecct_color(count):
    """
    selecct_color picks random colors

    Parameters
    ----------
    count : int
        the number of colors
    """
    return (
        random.choices(
            population=list(
                mcolors.CSS4_COLORS.keys()),
            k=count))


##################### Run: section1 for make_moel trends #################
if __name__ == "__main__":
    # set workspace
    workspace = os.path.dirname(
        os.path.abspath(
            inspect.getfile(
                inspect.currentframe())))
    # load the initial df
    init_df = pd.read_csv("./data/init_info_df.csv", index_col=0)
    init_df["make_model"] = init_df.make + " " + init_df.model
    # a list of unique make_models
    make_model_list = init_df.make_model.unique()
    # split items to lists which contains 5 item
    veh_arr = []
    for i in range(0, len(make_model_list) - 5, 5):
        veh_arr.append([make_model_list[i],
                        make_model_list[i + 1],
                        make_model_list[i + 2],
                        make_model_list[i + 3],
                        make_model_list[i + 4]])
    # run for each list
    df_trend = pd.DataFrame()
    factors = []
    for veh_list in veh_arr:
        pytrends_1 = TrendReq()
        # set date and the location for searching trends
        pytrends_1.build_payload(
            veh_list, geo='US', timeframe="2014-01-01 2020-08-30")
        df_1 = pytrends_1.interest_over_time()
        df_1 = df_1.iloc[:, :-1]
        average_1 = list(df_1.mean().values)
        factors.append(average_1[0])
        df_trend = pd.concat((df_trend, df_1), axis=1)
    # merge all data
    df_trend_final = pd.DataFrame()
    for column in np.unique(df_trend.columns):
        try:
            temp_df = pd.DataFrame(
                df_trend[column].mean(
                    axis=1), columns=[column])
        except BaseException:
            temp_df = pd.DataFrame(df_trend[column], columns=[column])
        df_trend_final = pd.concat((df_trend_final, temp_df), axis=1)
    # create a dictionary for colors
    plt_dict = dict(zip(list(df_trend_final.columns),
                        selecct_color(len(df_trend_final.columns))))
    df_trend_final = df_trend_final.rename_axis("date").reset_index()
    # plot
    trends_line_plot(
        df_trend_final,
        "date",
        plt_dict,
        xlabel="date",
        ylabel="count",
        file_name="google_trend",
        figureSize=(
            20,
            10),
        path=workspace)
    # save the final df
    df_trend_final.to_csv("./make_model_googlescore.csv")
