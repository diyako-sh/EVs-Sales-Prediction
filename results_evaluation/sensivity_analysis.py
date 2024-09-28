# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import pandas as pd
from dateutil.relativedelta import *
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from math import *
import keras
from res_att_lstm_functions import normalizer_func
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.figure(figsize=(10, 100))
pyplot.style.use('dark_background')
plt.style.use('dark_background')
################################ Functions #################################


def hybrid_model_sens_analysis(
        v_make,
        v_model,
        date_,
        input_df,
        past_history=7):
    """
    hybrid_model_sens_analysis: checks the sensivity-analysis of the models.

    Parameters
    ----------
    v_make : str
        vehicle make
    v_model : str
        vehicle model
    date_ : str
        date
    input_df : dataframe
        the input dataframe
    past_history : int
        the size of look-back window, by default 7

    """
    # make dummy month variables
    for i in range(1, 12):
        input_df['month_' + str(i)] = 0
        input_df['month_' + str(i)][input_df.date.dt.month == i] = 1
    # select features
    features_considered = [
        'sales',
        'shoppers',
        'month_1',
        'month_2',
        'month_3',
        'month_4',
        'month_5',
        'month_6',
        'month_7',
        'month_8',
        'month_9',
        'month_10',
        'month_11',
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
        'warranty_year']
    # data preprocessing
    features = input_df[features_considered]
    features.index = input_df['date']
    train_split = round(0.7 * len(features))
    dataset1, df_mean1, df_sd1, max_train_sale = normalizer_func(
        features, train_split)
    dataset2 = pd.DataFrame(dataset1)
    dataset2.columns = features_considered
    dataset2.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset2 = dataset2.dropna(axis=1)
    df_sd2 = []
    df_mean2 = []
    for n in range(len(df_sd1)):
        if df_sd1[n] != 0:
            df_sd2.append(df_sd1[n])
            df_mean2.append(df_mean1[n])
        else:
            pass
    df_sd = np.array(df_sd2)
    df_mean = np.array(df_mean2)
    del dataset1, df_sd1, df_sd2, df_mean1, df_mean2
    # load the seved model
    single_step_model = keras.models.load_model(
        f"./LSTM/{v_make}_{v_model}_{date_}_model.keras")
    # sensivity analysis step
    sen_feat = []
    sen_min = []
    sen_q1 = []
    sen_q2 = []
    sen_q3 = []
    sen_max = []
    val_min = []
    val_q1 = []
    val_q2 = []
    val_q3 = []
    val_max = []
    feat_data2 = features.iloc[:train_split, :]
    feat_data2 = feat_data2[dataset2.columns]
    sen_feat.append(feat_data2.columns[0])
    # pick the sale column for checking
    col = feat_data2.iloc[:, 0:1]
    # pick some vlues of train data: min, q1, q2, q3, max values
    min_val = col.min() / max_train_sale
    q1 = float(col.quantile(0.25).values / max_train_sale)
    q2 = float(col.quantile(0.5).values / max_train_sale)
    q3 = float(col.quantile(0.75).values / max_train_sale)
    max_val = col.max() / max_train_sale
    # define a place holder
    zero_df = pd.DataFrame(np.zeros((past_history, feat_data2 .shape[1])))
    # fill by min values and make a prediction
    min_mat = zero_df.copy()
    min_mat.iloc[:, 0:1] = min_val
    sale1 = single_step_model.predict(
        min_mat.values.reshape(
            1,
            min_mat.values.shape[0],
            min_mat.shape[1]))
    sen_min.append(float(sale1 * max_train_sale))
    val_min.append(float(col.min()))
    # fill by q1 values and make a prediction
    q1_mat = zero_df.copy()
    q1_mat.iloc[:, 0:1] = q1
    sale2 = single_step_model.predict(
        q1_mat.values.reshape(
            1,
            q1_mat.values.shape[0],
            q1_mat.shape[1]))
    sen_q1.append(float(sale2 * max_train_sale))
    val_q1.append(float(col.quantile(0.25).values))
    # fill by q2 values and make a prediction
    q2_mat = zero_df.copy()
    q2_mat.iloc[:, 0:1] = q2
    sale3 = single_step_model.predict(
        q2_mat.values.reshape(
            1,
            q2_mat.values.shape[0],
            q2_mat.shape[1]))
    sen_q2.append(float(sale3 * max_train_sale))
    val_q2.append(float(col.quantile(0.5).values))
    # fill by q3 values and make a prediction
    q3_mat = zero_df.copy()
    q3_mat.iloc[:, 0:1] = q3
    sale4 = single_step_model.predict(
        q3_mat.values.reshape(
            1,
            q3_mat.values.shape[0],
            q3_mat.shape[1]))
    sen_q3.append(float(sale4 * max_train_sale))
    val_q3.append(float(col.quantile(0.75).values))
    # fill by max values and make a prediction
    max_mat = zero_df.copy()
    max_mat.iloc[:, 0:1] = max_val
    sale5 = single_step_model.predict(
        max_mat.values.reshape(
            1,
            max_mat.values.shape[0],
            max_mat.shape[1]))
    sen_max.append(float(sale5 * max_train_sale))
    val_max.append(float(col.max()))
    # summarize results in a dataframe
    pd.DataFrame({'featuer': sen_feat,
                  'min_pred': sen_min,
                  'q1_pred': sen_q1,
                  'q2_pred': sen_q2,
                  'q3_pred': sen_q3,
                  'max_pred': sen_max,
                  'val_min': val_min,
                  'val_q1': val_q1,
                  'val_q2': val_q2,
                  'val_q3': val_q3,
                  'val_max': val_max})
    # repeat this process for all features
    for i in range(1, feat_data2.shape[1]):
        sen_feat.append(feat_data2.columns[i])
        # pick a feature for checking
        col = feat_data2.iloc[:, i:i + 1]
        # pick some vlues of train data: min, q1, q2, q3, max values
        min_val = (col.min() - df_mean[i]) / df_sd[i]
        q1 = (float(col.quantile(0.25).values) - df_mean[i]) / df_sd[i]
        q2 = (float(col.quantile(0.5).values) - df_mean[i]) / df_sd[i]
        q3 = (float(col.quantile(0.75).values) - df_mean[i]) / df_sd[i]
        max_val = (col.max() - df_mean[i]) / df_sd[i]
        # fill by min values and make a prediction
        min_mat = zero_df.copy()
        min_mat.iloc[:, i:i + 1] = min_val
        sale1 = single_step_model.predict(
            min_mat.values.reshape(
                1, min_mat.values.shape[0], min_mat.shape[1]))
        sen_min.append(float(sale1 * max_train_sale))
        val_min.append(float(col.min()))
        # fill by q1 values and make a prediction
        q1_mat = zero_df.copy()
        q1_mat.iloc[:, i:i + 1] = q1
        sale2 = single_step_model.predict(
            q1_mat.values.reshape(
                1, q1_mat.values.shape[0], q1_mat.shape[1]))
        sen_q1.append(float(sale2 * max_train_sale))
        val_q1.append(float(col.quantile(0.25).values))
        # fill by q2 values and make a prediction
        q2_mat = zero_df.copy()
        q2_mat.iloc[:, i:i + 1] = q2
        sale3 = single_step_model.predict(
            q2_mat.values.reshape(
                1, q2_mat.values.shape[0], q2_mat.shape[1]))
        sen_q2.append(float(sale3 * max_train_sale))
        val_q2.append(float(col.quantile(0.5).values))
        # fill by q3 values and make a prediction
        q3_mat = zero_df.copy()
        q3_mat.iloc[:, i:i + 1] = q3
        sale4 = single_step_model.predict(
            q3_mat.values.reshape(
                1, q3_mat.values.shape[0], q3_mat.shape[1]))
        sen_q3.append(float(sale4 * max_train_sale))
        val_q3.append(float(col.quantile(0.75).values))
        # fill by max values and make a prediction
        max_mat = zero_df.copy()
        max_mat.iloc[:, i:i + 1] = max_val
        sale5 = single_step_model.predict(
            max_mat.values.reshape(
                1, max_mat.values.shape[0], max_mat.shape[1]))
        sen_max.append(float(sale5 * max_train_sale))
        val_max.append(float(col.max()))
    # summarize the results
    sen_df_final = pd.DataFrame({'featuer': sen_feat,
                                'min_pred': sen_min,
                                 'q1_pred': sen_q1,
                                 'q2_pred': sen_q2,
                                 'q3_pred': sen_q3,
                                 'max_pred': sen_max,
                                 'val_min': val_min,
                                 'val_q1': val_q1,
                                 'val_q2': val_q2,
                                 'val_q3': val_q3,
                                 'val_max': val_max})

    sen_df_final['sensivity_value'] = None
    # calculate differences between max values and min values according by
    # each item(min, q1, q2, q3, max)
    for j in range(len(sen_df_final)):
        sen_df_final.sensivity_value[j] = sen_df_final.iloc[j, 1:6].max(
        ) - sen_df_final.iloc[j, 1:6].min()
    # sort based on high differences
    sen_df_final = sen_df_final.sort_values(
        by=['sensivity_value'],
        ascending=False).reset_index(
        drop=True)
    # pick top 5
    sen_df_final = sen_df_final.iloc[:5, :]
    # save the results
    sen_df_final.to_csv(f"./LSTM/result/{date_}/{v_make}_{v_model}_sen_df.csv")
    # plot the results
    for z in range(len(sen_df_final)):
        fig = pyplot.figure(figsize=(18, 10))
        y_plot = sen_df_final.iloc[z, 1:6]
        x_plot = sen_df_final.iloc[z, 6:11]
        pyplot.plot(
            y_plot,
            marker='o',
            markersize=3,
            color="yellow",
            alpha=0.8)
        # pyplot.legend()
        pyplot.xlabel("values")
        pyplot.ylabel("pred_sales")
        pyplot.title(f'feature : {sen_df_final.featuer[z]}')
        pyplot.xticks(np.arange(5), x_plot, rotation=90)
        pyplot.savefig(
            f"./LSTM/result/{date_}/{v_make}_{v_model}_feat_{z + 1}.png")
        pyplot.clf()


################################ Run #################################
if __name__ == "__main__":
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
    # load input data
    df_type = pd.read_csv("./data/vehicles_type_df.csv", index_col=0)
    elec_vehicles = df_type[df_type["type"] != "g"].drop_duplicates(
        ["make", "model"])[["make", "model", "category"]].values[:]
    input_df = pd.read_csv("./data/final_input_data.csv", index_col=0)
    # data preprocessing
    input_df['date'] = pd.to_datetime(input_df["month"])
    input_df['month'] = input_df.date.apply(lambda x: x.month)
    input_df['year'] = input_df.date.apply(lambda x: x.year)
    pred_steps = 3
    max_date1 = pd.to_datetime(input_df.date.max())
    # specify date for testing
    start_date = max_date1 - relativedelta(months=15)
    date_lst = []
    for i in range(6, 14):
        date_lst.append(
            pd.to_datetime(start_date) +
            relativedelta(
                months=+
                i +
                1))
    miss_mm = []
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
        # run for EVs
        for mmc in elec_vehicles:
            v_make, v_model, category = mmc
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
                    (model_df.set_index("date"), date_range_df), axis=1).reset_index()
                model_df.loc[:, "month"] = model_df.date.apply(
                    lambda x: x.month)
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
                    hybrid_model_sens_analysis(
                        v_make,
                        v_model,
                        strf_date,
                        model_df).set_index("date").replace(
                        0,
                        0.00000001)
                except BaseException:
                    miss_mm.append([v_make, v_model])
            else:
                pass
