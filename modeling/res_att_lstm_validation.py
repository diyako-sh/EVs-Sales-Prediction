# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import pandas as pd
import numpy as np
import os
import warnings
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('dark_background')
warnings.filterwarnings('ignore')
#################################### Run ####################################
# make directory
try:
    os.mkdir('./LSTM/result/cross_val')
except BaseException:
    pass
# set seed
np.random.seed(42)
input_df = pd.read_csv("./data/final_input_data.csv", index_col=0)
input_df['date'] = pd.to_datetime(input_df["month"])
# define some lists for results
r2_score = []
sum_sale = []
# define some dataframes for results
df_first_pred = pd.DataFrame()
df_second_pred = pd.DataFrame()
df_third_pred = pd.DataFrame()
max_date1 = pd.to_datetime(input_df.date.max())
# specify date for testing
start_date = max_date1 - relativedelta(months=15)
date_lst = []
for i in range(1, 14):
    date_lst.append(pd.to_datetime(start_date) + relativedelta(months=+i + 1))
for date in date_lst:
    strf_date = date.strftime("%Y_%m_%d")
    # load model results for the first presdictions
    df1 = pd.read_csv("./LSTM/result/" + strf_date +
                      "/sales_firs_predict.csv").drop(columns="Unnamed: 0")
    df_first_pred = pd.concat([df_first_pred, df1])
    # load model results for the second presdictions
    df2 = pd.read_csv("./LSTM/result/" + strf_date +
                      "/sales_second_predict.csv").drop(columns="Unnamed: 0")
    df_second_pred = pd.concat([df_second_pred, df2])
    # load model results for the third presdictions
    df3 = pd.read_csv("./LSTM/result/" + strf_date +
                      "/sales_third_predict.csv").drop(columns="Unnamed: 0")
    df_third_pred = pd.concat([df_third_pred, df3])
# .    
### _____________________ Step1: check sales of vehicles _____________________###
# add real sales values to the dataframe containing the first month predictions
df_first_pred = df_first_pred.reset_index().drop(columns="index")
df_first_pred["date"] = pd.to_datetime(df_first_pred["date"])
df_first_pred["real_sale"] = None
for i in range(len(df_first_pred)):
    v_make = df_first_pred.make_model[i].split("_")[0]
    v_model = df_first_pred.make_model[i].split("_")[1]
    v_date = df_first_pred.date[i]
    try:
        df_first_pred.real_sale[i] = input_df[input_df.make == v_make][input_df.model == v_model][input_df.date == v_date].sales.values[0]
    except BaseException:
        df_first_pred.real_sale[i] = None
# clean and sort dataframe
df_first_pred = df_first_pred.dropna(axis=0)
df_first_pred = df_first_pred.sort_values(["make_model", "date"])
df_first_pred = df_first_pred.reset_index(drop=True)
df_first_pred.to_csv("./LSTM/result/cross_val/df_validation_first_pred.csv")
# scatter plots for results
y_pred_first_pred = df_first_pred['sales_predict']
y_test_first_pred = df_first_pred['real_sale']
plt.scatter(y_test_first_pred, y_pred_first_pred)
plt.xlabel('real_sales')
plt.ylabel('predicted_sales')
plt.title('first_predict')
plt.savefig('./LSTM/result/cross_val/first_pred_sales.png')
plt.clf()
# calculate the R2 score according to the linear regression fitted on the
# scatter plot
y_pred_first_pred = sm.add_constant(y_pred_first_pred)
reg_model1 = sm.OLS(
    y_test_first_pred.astype(float),
    y_pred_first_pred.astype(float)).fit()
r2_score.append(reg_model1.rsquared)
sum_sale.append(df_first_pred.real_sale.sum())
# add real sales values to the dataframe containing the second month
# predictions
df_second_pred = df_second_pred.reset_index().drop(columns="index")
df_second_pred["date"] = pd.to_datetime(df_second_pred["date"])
df_second_pred["real_sale"] = None
for i in range(len(df_second_pred)):
    v_make = df_second_pred.make_model[i].split("_")[0]
    v_model = df_second_pred.make_model[i].split("_")[1]
    v_date = df_second_pred.date[i]
    try:
        df_second_pred.real_sale[i] = input_df[input_df.make ==v_make][input_df.model == v_model][input_df.date == v_date].sales.values[0]
    except BaseException:
        df_second_pred.real_sale[i] = None
# clean and sort data
df_second_pred = df_second_pred.dropna(axis=0)
df_second_pred = df_second_pred.sort_values(["make_model", "date"])
df_second_pred = df_second_pred.reset_index(drop=True)
df_second_pred.to_csv("./LSTM/result/cross_val/df_validation_second_pred.csv")
# scatter plots for results
y_pred_second_pred = df_second_pred['sales_predict']
y_test_second_pred = df_second_pred['real_sale']
plt.scatter(y_test_second_pred, y_pred_second_pred)
plt.xlabel('real_sales')
plt.ylabel('predicted_sales')
plt.title('second_predict')
plt.savefig('./LSTM/result/cross_val/second_pred_sales.png')
plt.clf()
# calculate the R2 score according to the linear regression fitted on the
# scatter plot
y_pred_second_pred = sm.add_constant(y_pred_second_pred)
reg_model2 = sm.OLS(
    y_test_second_pred.astype(float),
    y_pred_second_pred.astype(float)).fit()
r2_score.append(reg_model2.rsquared)
sum_sale.append(df_second_pred.real_sale.sum())
# add real sales values to the dataframe containing the third month predictions
df_third_pred = df_third_pred.reset_index().drop(columns="index")
df_third_pred["date"] = pd.to_datetime(df_third_pred["date"])
df_third_pred["real_sale"] = None
for i in range(len(df_third_pred)):
    v_make = df_third_pred.make_model[i].split("_")[0]
    v_model = df_third_pred.make_model[i].split("_")[1]
    v_date = df_third_pred.date[i]
    try:
        df_third_pred.real_sale[i] = input_df[input_df.make == v_make][input_df.model == v_model][input_df.date == v_date].sales.values[0]
    except BaseException:
        df_third_pred.real_sale[i] = None
# clean and sort dataframe
df_third_pred = df_third_pred.dropna(axis=0)
df_third_pred = df_third_pred.sort_values(["make_model", "date"])
df_third_pred = df_third_pred.reset_index(drop=True)
df_third_pred.to_csv("./LSTM/result/cross_val/df_validation_third_pred.csv")
# scatter plots for results
y_pred_third_pred = df_third_pred['sales_predict']
y_test_third_pred = df_third_pred['real_sale']
plt.scatter(y_test_third_pred, y_pred_third_pred)
plt.xlabel('real_sales')
plt.ylabel('predicted_sales')
plt.title('third_predict')
plt.savefig('./LSTM/result/cross_val/third_pred_sales.png')
plt.clf()
# calculate the R2 score according to the linear regression fitted on the
# scatter plot
y_pred_third_pred = sm.add_constant(y_pred_third_pred)
reg_model3 = sm.OLS(
    y_test_third_pred.astype(float),
    y_pred_third_pred.astype(float)).fit()
r2_score.append(reg_model3.rsquared)
sum_sale.append(df_third_pred.real_sale.sum())
# summaraize the results
r2_df = pd.DataFrame({'pred_level': [1, 2, 3],
                      'r2_score': r2_score,
                      'sum_real_sale': sum_sale})
r2_df.to_csv('./LSTM/result/r2_df_sales.csv')
# .
### _____________________ Step2: check shares of vehicles _____________________###
# check the scatter plot and R2 score for shares of vehicles
# for the first month predictions
r2_score_share = []
sum_share = []
share_df_first = pd.DataFrame()
df_first_pred["date"] = pd.to_datetime(df_first_pred["date"])
months = df_first_pred.date.drop_duplicates()
# calculate the share in each month
for m in range(len(months)):
    month_df = df_first_pred[df_first_pred.date == months[m]]
    month_df['pred_share'] = month_df['sales_predict'] / month_df.sales_predict.sum()
    month_df['real_share'] = month_df['real_sale'] / month_df.real_sale.sum()
    share_df_first = pd.concat([share_df_first, month_df], axis=0)
share_df_first.to_csv("./LSTM/result/cross_val/df_validation_first_share.csv")
# scatter plot for shares
share_pred_first_pred = share_df_first['pred_share']
share_test_first_pred = share_df_first['real_share']
plt.scatter(share_pred_first_pred, share_test_first_pred)
plt.xlabel('real_share')
plt.ylabel('predicted_share')
plt.title('first_predict')
plt.savefig('./LSTM/result/cross_val/first_pred_share.png')
plt.clf()
# calculate the R2 score according to the linear regression fitted on the
# shares scatter plot
share_pred_first_pred = sm.add_constant(share_pred_first_pred)
reg_model5 = sm.OLS(
    share_test_first_pred.astype(float),
    share_pred_first_pred.astype(float)).fit()
r2_score_share.append(reg_model5.rsquared)
sum_share.append(share_df_first.real_share.sum())
# for the second month predictions
share_df_second = pd.DataFrame()
df_second_pred["date"] = pd.to_datetime(df_second_pred["date"])
months = df_second_pred.date.drop_duplicates()
# calculate the share in each month
for m in range(len(months)):
    month_df = df_second_pred[df_second_pred.date == months[m]]
    month_df['pred_share'] = month_df['sales_predict'] / month_df.sales_predict.sum()
    month_df['real_share'] = month_df['real_sale'] / month_df.real_sale.sum()
    share_df_second = pd.concat([share_df_second, month_df], axis=0)
share_df_second.to_csv(
    "./LSTM/result/cross_val/df_validation_second_share.csv")
# scatter plot for shares
share_pred_second_pred = share_df_second['pred_share']
share_test_second_pred = share_df_second['real_share']
plt.scatter(share_pred_second_pred, share_test_second_pred)
plt.xlabel('real_share')
plt.ylabel('predicted_share')
plt.title('second_predict')
plt.savefig('./LSTM/result/cross_val/second_pred_share.png')
plt.clf()
# calculate the R2 score according to the linear regression fitted on the
# shares scatter plot
share_pred_second_pred = sm.add_constant(share_pred_second_pred)
reg_model6 = sm.OLS(
    share_test_second_pred.astype(float),
    share_pred_second_pred .astype(float)).fit()
r2_score_share.append(reg_model6.rsquared)
sum_share.append(share_df_second.real_share.sum())
# for the third month predictions
share_df_third = pd.DataFrame()
df_third_pred["date"] = pd.to_datetime(df_third_pred["date"])
months = df_third_pred.date.drop_duplicates()
# calculate the share in each month
for m in range(len(months)):
    month_df = df_third_pred[df_third_pred.date == months[m]]
    month_df['pred_share'] = month_df['sales_predict'] / month_df.sales_predict.sum()
    month_df['real_share'] = month_df['real_sale'] / month_df.real_sale.sum()
    share_df_third = pd.concat([share_df_third, month_df], axis=0)
share_df_third.to_csv("./LSTM/result/cross_val/df_validation_third_share.csv")
# scatter plot for shares
share_pred_third_pred = share_df_third['pred_share']
share_test_third_pred = share_df_third['real_share']
plt.scatter(share_pred_third_pred, share_test_third_pred)
plt.xlabel('real_share')
plt.ylabel('predicted_share')
plt.title('third_predict')
plt.savefig('./LSTM/result/cross_val/third_pred_share.png')
plt.clf()
# calculate the R2 score according to the linear regression fitted on the
# shares scatter plot
share_pred_third_pred = sm.add_constant(share_pred_third_pred)
reg_model7 = sm.OLS(
    share_test_third_pred.astype(float),
    share_pred_third_pred .astype(float)).fit()
r2_score_share.append(reg_model7.rsquared)
sum_share.append(share_df_third.real_share.sum())
# summarize and save final results
r2_df_share = pd.DataFrame({'pred_level': [1, 2, 3],
                            'r2_score': r2_score_share,
                            'sum_real_sale': sum_share})
r2_df_share.to_csv('./LSTM/result/r2_df_shares.csv')
