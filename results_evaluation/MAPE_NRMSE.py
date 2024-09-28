# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import pandas as pd
import numpy as np
from dateutil.relativedelta import *
import math
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from math import *
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 100))
plt.style.use('dark_background')
################################ Functions #################################


def mape(y_real, y_pred):
    """
    mape calculates the Mean Absolute Percentage Error.

    Parameters
    ----------
    y_real : pandas.series
        real values
    y_pred : pandas.series
        predicted values

    Returns
    -------
    mape_score : float
        the Mean Absolute Percentage Error
    """
    mape_score = np.mean(np.abs((y_real - y_pred) / y_real))
    return mape_score


def nrmse_range(y_real, y_pred):
    """
    nrmse_range calculates the Root Mean Square Error normalized by the change range.

    Parameters
    ----------
    y_real : pandas.series
        real values
    y_pred : pandas.series
        predicted values

    Returns
    -------
    nrmse : float
        the Root Mean Square Error normalized by the change range
    """
    mse = np.square(np.subtract(y_real, y_pred)).mean()
    rmse = math.sqrt(mse)
    nrmse = rmse / (y_real.max() - y_real.min())
    return nrmse


def nrmse_mean(y_real, y_pred):
    """
    nrmse_mean calculates the Root Mean Square Error normalized by the mean value.

    Parameters
    ----------
    y_real : pandas.series
        real values
    y_pred : pandas.series
        predicted values

    Returns
    -------
    nrmse
        the Root Mean Square Error normalized by the mean value
    """
    mse = np.square(np.subtract(y_real, y_pred)).mean()
    rmse = math.sqrt(mse)
    nrmse = rmse / y_real.mean()
    return nrmse


################################ Run #################################
if __name__ == "__main__":
    r2_score = []
    sum_sale = []
    # .
    # ____________ Stage1: check the first month's prediction ____________###
    df_first_pred = pd.read_csv(
        "./LSTM/result/cross_val/df_validation_first_pred.csv",
        index_col=0)
    y_pred_first_pred = df_first_pred['sales_predict']
    y_test_first_pred = df_first_pred['real_sale']
    # calculate the R2 score according to the linear regression fitted on
    # predicted&real values
    y_pred_first_pred = sm.add_constant(y_pred_first_pred)
    reg_model1 = sm.OLS(
        y_test_first_pred.astype(float),
        y_pred_first_pred.astype(float)).fit()
    r2_score.append(reg_model1.rsquared)
    sum_sale.append(df_first_pred.real_sale.sum())
    data1 = df_first_pred.copy()
    make = data1.drop_duplicates(
        ["make_model"]).make_model.reset_index(drop=True)
    make_model = []
    sum_sales = []
    mape_lst = []
    nrmse_range_lst = []
    nrmse_mean_lst = []
    # repeat for all vehicles
    for i in range(len(make)):
        df = data1[data1.make_model == make[i]]
        if len(df) > 1:
            make_model.append(make[i])
            sum_sales.append(df.real_sale.sum())
            # calculate the mape
            mape_lst.append(mape(df.real_sale, df.sales_predict))
            # calculate the nrmse_range
            nrmse_range_lst.append(nrmse_range(df.real_sale, df.sales_predict))
            # calculate the nrmse_mean
            nrmse_mean_lst.append(nrmse_mean(df.real_sale, df.sales_predict))
        else:
            pass
    # summarize and save the results
    df_first_pred = pd.DataFrame({"make_model": make_model,
                                  "sum_sales": sum_sales,
                                  "mape": mape_lst,
                                  "nrmse_range": nrmse_range_lst,
                                  "nrmse_mean": nrmse_mean_lst})
    df_first_pred.to_csv("./LSTM/result/cross_val/sum_result_first_pred.csv")
    first_pred_weihted_mean = sm.stats.DescrStatsW(
        df_first_pred.drop(
            columns=[
                "make_model",
                "sum_sales"]),
        weights=df_first_pred.sum_sales).mean
    first_pred_mean = list(
        df_first_pred.drop(
            columns=[
                "make_model",
                "sum_sales"]).mean())
    # plot the results for each vehicle
    for i in range(len(make)):
        df = data1[data1.make_model == make[i]]
        df.index = df.date
        plt.figure(figsize=(14, 7))
        plt.plot(df.sales_predict, label='sales_predict', c="lime")
        plt.plot(df.real_sale, label='real_sale', c="magenta")
        plt.legend()
        plt.title(f"{make[i]}")
        plt.xlabel("date")
        plt.ylabel("sales")
        plt.savefig(f"./LSTM/result/cross_val/{make[i]}_first_pred_trend.png")
        plt.clf()
    # scatter plot for all results
    y_pred_first_pred = data1['sales_predict']
    y_test_first_pred = data1['real_sale']
    plt.scatter(y_test_first_pred, y_pred_first_pred)
    plt.xlabel('real_sales')
    plt.ylabel('predicted_sales')
    plt.title('first_predict')
    plt.savefig('./LSTM/result/cross_val/first_pred_scatter.png')
    plt.clf()
    # .
    # ____________ Stage2: check the second month's prediction ____________###
    df_second_pred = pd.read_csv(
        "./LSTM/result/cross_val/df_validation_second_pred.csv",
        index_col=0)
    y_pred_second_pred = df_second_pred['sales_predict']
    y_test_second_pred = df_second_pred['real_sale']
    # calculate the R2 score according to the linear regression fitted on
    # predicted&real values
    y_pred_second_pred = sm.add_constant(y_pred_second_pred)
    reg_model2 = sm.OLS(
        y_test_second_pred.astype(float),
        y_pred_second_pred.astype(float)).fit()
    r2_score.append(reg_model2.rsquared)
    sum_sale.append(df_second_pred.real_sale.sum())
    data2 = df_second_pred.copy()
    make = data2.drop_duplicates(
        ["make_model"]).make_model.reset_index(
        drop=True)
    make_model = []
    sum_sales = []
    mape_lst = []
    nrmse_range_lst = []
    nrmse_mean_lst = []
    # repeat for all vehicles
    for i in range(len(make)):
        df = data2[data2.make_model == make[i]]
        if len(df) > 1:
            make_model.append(make[i])
            sum_sales.append(df.real_sale.sum())
            # calculate the mape
            mape_lst.append(mape(df.real_sale, df.sales_predict))
            # calculate the nrmse_range
            nrmse_range_lst.append(nrmse_range(df.real_sale, df.sales_predict))
            # calculate the nrmse_mean
            nrmse_mean_lst.append(nrmse_mean(df.real_sale, df.sales_predict))
        else:
            pass
        # summarize and save the results
    df_second_pred = pd.DataFrame({"make_model": make_model,
                                   "sum_sales": sum_sales,
                                   "mape": mape_lst,
                                   "nrmse_range": nrmse_range_lst,
                                   "nrmse_mean": nrmse_mean_lst})
    df_second_pred.to_csv("./LSTM/result/cross_val/sum_result_second_pred.csv")
    second_pred_weihted_mean = sm.stats.DescrStatsW(
        df_second_pred.drop(
            columns=[
                "make_model",
                "sum_sales"]),
        weights=df_second_pred.sum_sales).mean
    second_pred_mean = list(
        df_second_pred.drop(
            columns=[
                "make_model",
                "sum_sales"]).mean())
    # plot the results for each vehicle
    for i in range(len(make)):
        df = data2[data2.make_model == make[i]]
        df.index = df.date
        plt.figure(figsize=(14, 7))
        plt.plot(df.sales_predict, label='sales_predict', c="lime")
        plt.plot(df.real_sale, label='real_sale', c="magenta")
        plt.legend()
        plt.title(f"{make[i]}")
        plt.xlabel("date")
        plt.ylabel("sales")
        plt.savefig(f"./LSTM/result/cross_val/{make[i]}_second_pred_trend.png")
        plt.clf()
    # scatter plot for all results
    y_pred_second_pred = data2['sales_predict']
    y_test_second_pred = data2['real_sale']
    plt.scatter(y_test_second_pred, y_pred_second_pred)
    plt.xlabel('real_sales')
    plt.ylabel('predicted_sales')
    plt.title('second_predict')
    plt.savefig('./LSTM/result/cross_val/second_pred_scatter.png')
    plt.clf()
    # .
    # ____________ Stage3: check the third month's prediction____________###
    df_third_pred = pd.read_csv(
        "./LSTM/result/cross_val/df_validation_third_pred.csv",
        index_col=0)
    y_pred_third_pred = df_third_pred['sales_predict']
    y_test_third_pred = df_third_pred['real_sale']
    # calculate the R2 score according to the linear regression fitted on
    # predicted&real values
    y_pred_third_pred = sm.add_constant(y_pred_third_pred)
    reg_model3 = sm.OLS(
        y_test_third_pred.astype(float),
        y_pred_third_pred.astype(float)).fit()
    r2_score.append(reg_model3.rsquared)
    sum_sale.append(df_third_pred.real_sale.sum())
    data3 = df_third_pred.copy()
    make = data3.drop_duplicates(
        ["make_model"]).make_model.reset_index(
        drop=True)
    make_model = []
    sum_sales = []
    mape_lst = []
    nrmse_range_lst = []
    nrmse_mean_lst = []
    # repeat for all vehicles
    for i in range(len(make)):
        df = data3[data3.make_model == make[i]]
        if len(df) > 1:
            make_model.append(make[i])
            sum_sales.append(df.real_sale.sum())
            # calculate the mape
            mape_lst.append(mape(df.real_sale, df.sales_predict))
            # calculate the nrmse_range
            nrmse_range_lst.append(nrmse_range(df.real_sale, df.sales_predict))
            # calculate the nrmse_mean
            nrmse_mean_lst.append(nrmse_mean(df.real_sale, df.sales_predict))
        else:
            pass
    # summarize and save the results
    df_third_pred = pd.DataFrame({"make_model": make_model,
                                  "sum_sales": sum_sales,
                                  "mape": mape_lst,
                                  "nrmse_range": nrmse_range_lst,
                                  "nrmse_mean": nrmse_mean_lst})
    df_third_pred.to_csv("./LSTM/result/cross_val/sum_result_third_pred.csv")
    third_pred_weihted_mean = sm.stats.DescrStatsW(
        df_third_pred.drop(
            columns=[
                "make_model",
                "sum_sales"]),
        weights=df_third_pred.sum_sales).mean
    third_pred_mean = list(
        df_third_pred.drop(
            columns=[
                "make_model",
                "sum_sales"]).mean())
    # plot the results for each vehicle
    for i in range(len(make)):
        df = data3[data3.make_model == make[i]]
        df.index = df.date
        plt.figure(figsize=(14, 7))
        plt.plot(df.sales_predict, label='sales_predict', c="lime")
        plt.plot(df.real_sale, label='real_sale', c="magenta")
        plt.legend()
        plt.title(f"{make[i]}")
        plt.xlabel("date")
        plt.ylabel("sales")
        plt.savefig(f"./LSTM/result/cross_val/{make[i]}_third_pred_trend.png")
        plt.clf()
    # scatter plot for all results
    y_pred_third_pred = data3['sales_predict']
    y_test_third_pred = data3['real_sale']
    plt.scatter(y_test_third_pred, y_pred_third_pred)
    plt.xlabel('real_sales')
    plt.ylabel('predicted_sales')
    plt.title('third_predict')
    plt.savefig('./LSTM/result/cross_val/third_pred_scatter.png')
    plt.clf()
    # summarize all the results
    # sales_result
    r2_df = pd.DataFrame({'pred_level': [1, 2, 3],
                          'r2_score': r2_score,
                          'sum_real_sale': sum_sale})
    r2_df.to_csv('./LSTM/result/r2_df_sales.csv')
    pred_level = []
    mean_mape = []
    mean_nrmse_range = []
    mean_nrmse_mean = []
    weighted_mean_mape = []
    weighted_mean_nrmse_range = []
    weighted_mean_nrmse_mean = []
    # results for the first month's prediction
    pred_level.append("first_pred")
    mean_mape.append(round(first_pred_mean[0], 3))
    mean_nrmse_range.append(round(first_pred_mean[1], 3))
    mean_nrmse_mean.append(round(first_pred_mean[2], 3))
    weighted_mean_mape.append(round(first_pred_weihted_mean[0], 3))
    weighted_mean_nrmse_range.append(round(first_pred_weihted_mean[1], 3))
    weighted_mean_nrmse_mean.append(round(first_pred_weihted_mean[2], 3))
    # results for the second month's prediction
    pred_level.append("second_pred")
    mean_mape.append(round(second_pred_mean[0], 3))
    mean_nrmse_range.append(round(second_pred_mean[1], 3))
    mean_nrmse_mean.append(round(second_pred_mean[2], 3))
    weighted_mean_mape.append(round(second_pred_weihted_mean[0], 3))
    weighted_mean_nrmse_range.append(round(second_pred_weihted_mean[1], 3))
    weighted_mean_nrmse_mean.append(round(second_pred_weihted_mean[2], 3))
    # results for the third month's prediction
    pred_level.append("third_pred")
    mean_mape.append(round(third_pred_mean[0], 3))
    mean_nrmse_range.append(round(third_pred_mean[1], 3))
    mean_nrmse_mean.append(round(third_pred_mean[2], 3))
    weighted_mean_mape.append(round(third_pred_weihted_mean[0], 3))
    weighted_mean_nrmse_range.append(round(third_pred_weihted_mean[1], 3))
    weighted_mean_nrmse_mean.append(round(third_pred_weihted_mean[2], 3))
    # generate and the final results dataframe
    final_result = pd.DataFrame({"pred_level": pred_level,
                                 "mean_mape": mean_mape,
                                 "mean_nrmse_range": mean_nrmse_range,
                                 "mean_nrmse_mean": mean_nrmse_mean,
                                 "weighted_mean_mape": weighted_mean_mape,
                                 "weighted_mean_nrmse_range": weighted_mean_nrmse_range,
                                 "weighted_mean_nrmse_mean": weighted_mean_nrmse_mean})
    final_result.to_csv("./LSTM/result/final_result_all_pred.csv")
