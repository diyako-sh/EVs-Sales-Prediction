# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import pandas as pd
import os
from res_att_lstm_functions import *
from dateutil.relativedelta import relativedelta
from random import seed, randint
from matplotlib import pyplot
pyplot.style.use('dark_background')
#################################### Run ####################################
# make directory
try:
    os.mkdir("./LSTM")
    os.mkdir("./LSTM/data")
    os.mkdir("./LSTM/result")
    os.mkdir("./LSTM/plots")
    os.mkdir("./LSTM/tuner")
except BaseException:
    try:
        os.mkdir("./LSTM/data")
        os.mkdir("./LSTM/result")
        os.mkdir("./LSTM/plots")
        os.mkdir("./LSTM/tuner")
    except BaseException:
        try:
            os.mkdir("./LSTM/result")
            os.mkdir("./LSTM/plots")
            os.mkdir("./LSTM/tuner")
        except BaseException:
            try:
                os.mkdir("./LSTM/plots")
                os.mkdir("./LSTM/tuner")
            except BaseException:
                try:
                    os.mkdir("./LSTM/tuner")
                except BaseException:
                    pass
# data preprocessing
input_df = pd.read_csv("./data/final_input_data.csv", index_col=0)
input_df['date'] = pd.to_datetime(input_df["month"])
input_df['month'] = input_df.date.apply(lambda x: x.month)
input_df['year'] = input_df.date.apply(lambda x: x.year)
pred_steps = 3
max_date1 = pd.to_datetime(input_df.date.max())
# specify date for testing
start_date = max_date1 - relativedelta(months=15)
date_lst = []
for i in range(2, 14):
    date_lst.append(pd.to_datetime(start_date) + relativedelta(months=+i + 1))
miss_mm = []
# run the model
for date in date_lst:
    make_model_list1 = []
    date_list1 = []
    forecast_list1 = []
    make_model_list2 = []
    date_list2 = []
    forecast_list2 = []
    make_model_list3 = []
    date_list3 = []
    forecast_list3 = []
    data = input_df[input_df.date < date]
    f_name = "./LSTM/data/" + date.strftime("%Y_%m_%d") + ".csv"
    data.to_csv(f_name)
    strf_date = date.strftime("%Y_%m_%d")
    # make folders for results
    try:
        os.mkdir("./LSTM/result/" + strf_date)
        os.mkdir("./LSTM/result/" + strf_date + "/detailes")
    except BaseException:
        try:
            os.mkdir("./LSTM/result/" + strf_date + "/detailes")
        except BaseException:
            pass
    model_data = data
    model_data["date"] = pd.to_datetime(model_data["date"])
    max_date = pd.to_datetime(model_data.date.max())
    inputs1 = model_data.drop_duplicates(["make", "model"])[
        ["make", "model", "category"]].reset_index().drop(columns='index')
    # seed random number generator
    seed(41)
    # generate 15 random indices
    random_inds = []
    for _ in range(15):
        random_inds.append(randint(0, len(inputs1)))
    # pick random vehicles
    inputs2 = inputs1.iloc[random_inds]
    inputs = inputs2.values[:]
    inputs = model_data.drop_duplicates(["make", "model"])[
        ["make", "model", "category"]].values[:]
    for mmc in inputs:
        v_make, v_model, category = mmc
        # select features
        model_df = model_data[model_data.make == v_make][model_data.model == v_model][['date',
                                                                                       'sales',
                                                                                       'shoppers',
                                                                                       'CCI',
                                                                                       'CPI',
                                                                                       'CPI_newvehicle',
                                                                                       'DowJones',
                                                                                       'Finance_rate_loan48m',
                                                                                       'Finance_rate_loan60m',
                                                                                       'GDP',
                                                                                       'personal_income_per_capita',
                                                                                       'PPI',
                                                                                       'SP500',
                                                                                       'make_news_score',
                                                                                       'make_model_news_score',
                                                                                       'make_model_gt_score',
                                                                                       'delaer_gt_score',
                                                                                       'price_gt_score',
                                                                                       'max_price',
                                                                                       'min_price',
                                                                                       'mean_price',
                                                                                       'max_mpg',
                                                                                       'min_mpg',
                                                                                       'mean_mpg',
                                                                                       'max_mileage',
                                                                                       'min_mileage',
                                                                                       'mean_mileage',
                                                                                       'max_engine_power',
                                                                                       'min_engine_power',
                                                                                       'mean_engine_power',
                                                                                       'max_safety_score',
                                                                                       'min_safety_score',
                                                                                       'mean_safety_score',
                                                                                       'max_options_score',
                                                                                       'min_options_score',
                                                                                       'mean_options_score',
                                                                                       'warranty_mile',
                                                                                       'warranty_year']].sort_values("date").reset_index(drop=True)
        # apply a filter on vehicles that don't have enough data
        if len(model_df) > 20:
            date_range_df = pd.DataFrame(
                pd.date_range(
                    model_df.date.min(),
                    max_date),
                columns=["date"])
            date_range_df.loc[:, "month"] = date_range_df.date.apply(
                lambda x: x.month)
            date_range_df.loc[:, "year"] = date_range_df.date.apply(
                lambda x: x.year)
            date_range_df = date_range_df.groupby(["year", "month"], as_index=False).agg({
                "date": min}).set_index("date")
            model_df = pd.concat(
                (model_df.set_index("date"),
                 date_range_df),
                axis=1).reset_index()
            model_df.loc[:, "month"] = model_df.date.apply(lambda x: x.month)
            model_df.loc[:, "year"] = model_df.date.apply(lambda x: x.year)
            # fill missing values with epsilon
            model_df.shoppers = model_df.shoppers.fillna(0.00000001)
            model_df.sales = model_df.sales.fillna(0.00000001)
            model_df = model_df.fillna(method="ffill")
            # group by month for monthly sale prediction
            model_df = model_df.groupby(["year",
                                         "month"],
                                        as_index=False).agg({'date': min,
                                                             'sales': sum,
                                                             'shoppers': sum,
                                                             'CCI': max,
                                                             'CPI': max,
                                                             'CPI_newvehicle': max,
                                                             'DowJones': max,
                                                             'Finance_rate_loan48m': max,
                                                             'Finance_rate_loan60m': max,
                                                             'GDP': max,
                                                             'personal_income_per_capita': max,
                                                             'PPI': max,
                                                             'SP500': max,
                                                             'make_news_score': max,
                                                             'make_model_news_score': max,
                                                             'make_model_gt_score': max,
                                                             'delaer_gt_score': max,
                                                             'price_gt_score': max,
                                                             'max_price': max,
                                                             'min_price': max,
                                                             'mean_price': max,
                                                             'max_mpg': max,
                                                             'min_mpg': max,
                                                             'mean_mpg': max,
                                                             'max_mileage': max,
                                                             'min_mileage': max,
                                                             'mean_mileage': max,
                                                             'max_engine_power': max,
                                                             'min_engine_power': max,
                                                             'mean_engine_power': max,
                                                             'max_safety_score': max,
                                                             'min_safety_score': max,
                                                             'mean_safety_score': max,
                                                             'max_options_score': max,
                                                             'min_options_score': max,
                                                             'mean_options_score': max,
                                                             'warranty_mile': max,
                                                             'warranty_year': max}).sort_values("date").reset_index()[['date',
                                                                                                                       'sales',
                                                                                                                       'shoppers',
                                                                                                                       'CCI',
                                                                                                                       'CPI',
                                                                                                                       'CPI_newvehicle',
                                                                                                                       'DowJones',
                                                                                                                       'Finance_rate_loan48m',
                                                                                                                       'Finance_rate_loan60m',
                                                                                                                       'GDP',
                                                                                                                       'personal_income_per_capita',
                                                                                                                       'PPI',
                                                                                                                       'SP500',
                                                                                                                       'make_news_score',
                                                                                                                       'make_model_news_score',
                                                                                                                       'make_model_gt_score',
                                                                                                                       'delaer_gt_score',
                                                                                                                       'price_gt_score',
                                                                                                                       'max_price',
                                                                                                                       'min_price',
                                                                                                                       'mean_price',
                                                                                                                       'max_mpg',
                                                                                                                       'min_mpg',
                                                                                                                       'mean_mpg',
                                                                                                                       'max_mileage',
                                                                                                                       'min_mileage',
                                                                                                                       'mean_mileage',
                                                                                                                       'max_engine_power',
                                                                                                                       'min_engine_power',
                                                                                                                       'mean_engine_power',
                                                                                                                       'max_safety_score',
                                                                                                                       'min_safety_score',
                                                                                                                       'mean_safety_score',
                                                                                                                       'max_options_score',
                                                                                                                       'min_options_score',
                                                                                                                       'mean_options_score',
                                                                                                                       'warranty_mile',
                                                                                                                       'warranty_year']]
            # replace zero values with epsilon
            model_df["shoppers"].replace(0, 0.00000001, inplace=True)
            model_df["sales"].replace(0, 0.00000001, inplace=True)
            # pick features on order
            model_df = model_df[['date',
                                 'sales',
                                 'shoppers',
                                 'CCI',
                                 'CPI',
                                 'CPI_newvehicle',
                                 'DowJones',
                                 'Finance_rate_loan48m',
                                 'Finance_rate_loan60m',
                                 'GDP',
                                 'personal_income_per_capita',
                                 'PPI',
                                 'SP500',
                                 'make_news_score',
                                 'make_model_news_score',
                                 'make_model_gt_score',
                                 'delaer_gt_score',
                                 'price_gt_score',
                                 'max_price',
                                 'min_price',
                                 'mean_price',
                                 'max_mpg',
                                 'min_mpg',
                                 'mean_mpg',
                                 'max_mileage',
                                 'min_mileage',
                                 'mean_mileage',
                                 'max_engine_power',
                                 'min_engine_power',
                                 'mean_engine_power',
                                 'max_safety_score',
                                 'min_safety_score',
                                 'mean_safety_score',
                                 'max_options_score',
                                 'min_options_score',
                                 'mean_options_score',
                                 'warranty_mile',
                                 'warranty_year']]
            try:
                LSTM_model_df = hybrid_res_att_lstm_model(
                    v_make,
                    v_model,
                    strf_date,
                    model_df,
                    forecast_steps=pred_steps).set_index("date").replace(
                    0,
                    0.00000001)
                LSTM_model_df.to_csv(
                    "./LSTM/result/" +
                    strf_date +
                    "/detailes/" +
                    v_make +
                    "_" +
                    v_model +
                    ".csv")
                # for the first month predtions
                make_model_list1.append(v_make + "_" + v_model)
                # for the second month predtions
                make_model_list2.append(v_make + "_" + v_model)
                # for the third month predtions
                make_model_list3.append(v_make + "_" + v_model)
                # .
                # for the first month predtions
                date_list1.append(LSTM_model_df.index[-3])
                # for the second month predtions
                date_list2.append(LSTM_model_df.index[-2])
                # for the third month predtions
                date_list3.append(LSTM_model_df.index[-1])
                # .
                # for the first month predtions
                forecast_list1.append(LSTM_model_df.sales[-3])
                # for the second month predtions
                forecast_list2.append(LSTM_model_df.sales[-2])
                # for the third month predtions
                forecast_list3.append(LSTM_model_df.sales[-1])
            except BaseException:
                miss_mm.append([v_make, v_model])
        else:
            pass
    # generate forecaste dataframes
    forecaste_df1 = pd.DataFrame({"make_model": make_model_list1,
                                  "date": date_list1,
                                  "sales_predict": forecast_list1})
    forecaste_df2 = pd.DataFrame({"make_model": make_model_list2,
                                  "date": date_list2,
                                  "sales_predict": forecast_list2})
    forecaste_df3 = pd.DataFrame({"make_model": make_model_list3,
                                  "date": date_list3,
                                  "sales_predict": forecast_list3})
    # save dataframes
    forecaste_df1.to_csv(
        "./LSTM/result/" +
        "/" +
        strf_date +
        "/sales_firs_predict.csv")
    forecaste_df2.to_csv(
        "./LSTM/result/" +
        "/" +
        strf_date +
        "/sales_second_predict.csv")
    forecaste_df3.to_csv(
        "./LSTM/result/" +
        "/" +
        strf_date +
        "/sales_third_predict.csv")
