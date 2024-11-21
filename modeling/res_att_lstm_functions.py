# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from keras_tuner.tuners import RandomSearch
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras import backend as K
from matplotlib import pyplot
from custom_funcs import attention
pyplot.style.use('dark_background')
################################ Functions #################################


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
    """
    multivariate_data separates X data(features) and Y data(labels)

    Parameters
    ----------
    dataset : dataframe
        the input dataframe
    target : dataframe
        the target column of the input dataframe
    start_index : int
        start index
    end_index : int
        end index
    history_size : int
        the size of the look-back window
    target_size : int
        the size of the look-forward window for prediction
    single_step : bool, optional
       single step for prediction, by default False

    Returns
    -------
    x_data : array
        x data
    y_data : array
        y data
    """
    data = []
    labels = []
    # set start_index and end_index
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    # separate data
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, target_size)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    x_data = np.array(data)
    y_data = np.array(labels)
    return x_data, y_data


def normalizer_func(data, trainsplit):
    """
    normalizer_func normalizes features(used for feature scaling)

    Parameters
    ----------
    data : dataframe
        the input dataframe containing all features
    trainsplit : int
        the index associated with train test split

    Returns
    -------
    temp_out : array
        normalized data
    temp_mean : array
        mean of features
    temp_std : array
        standard deviation of features
    mx_sale : int
        the maximum value of sale data
    """
    temp = data.values
    # the first column is sales
    mx_sale = temp[:trainsplit, 0].max()
    # scaling
    temp_mean = temp[:trainsplit].mean(axis=0)
    temp_std1 = temp[:trainsplit].std(axis=0)
    temp_std2 = []
    for k in range(len(temp_std1)):
        if temp_std1[k] > 0.001:
            temp_std2.append(temp_std1[k])
        else:
            temp_std2.append(0)
    temp_std = np.array(temp_std2)
    temp_out = (temp - temp_mean) / temp_std
    temp_out[:, 0] = temp[:, 0] / mx_sale
    return temp_out, temp_mean, temp_std, mx_sale



def hybrid_res_att_lstm_model(
        v_make,
        v_model,
        date_,
        input_df,
        forecast_steps=3,
        past_history=7,
        future_steps=1,
        batch_size=250,
        buffer_size=1000,
        epochs_=100,
        object_index=0):
    """
    hybrid_res_att_lstm_model: The hybrid-residual-attention-LSTM model creates, trains, and automatically fine-tunes the model; then, makes predictions.

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
    forecast_steps : int
        the number of future months predicted in each prediction step, by default 3
    past_history : int
        the size of look-back window, by default 7
    future_steps : int
        the number of prediction steps in each run, by default 1
    batch_size : int
        batch size, by default 250
    buffer_size : int
        buffer size, by default 1000
    epochs_ : int
        the number of epochs, by default 100
    object_index : int
        the target columns, by default 0

    Returns
    -------
    final_df : dataframe
        a dataframe containing predictions
    """
    # make directory
    try:
        os.mkdir("./LSTM/tuner/" + date_)
        os.mkdir("./LSTM/plots/" + date_)
    except BaseException:
        try:
            os.mkdir("./LSTM/plots/" + date_)
        except BaseException:
            pass
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
    dataset = dataset2.to_numpy()
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
    # train_validation_test split
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, object_index], 0,
                                                       train_split, past_history,
                                                       future_steps, single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, object_index],
                                                   train_split, None, past_history,
                                                   future_steps, single_step=True)

    all_data_x, all_data_y = multivariate_data(dataset, dataset[:, object_index], 0,
                                               len(dataset) - 1, past_history,
                                               future_steps, single_step=True)
    train_data_single = tf.data.Dataset.from_tensor_slices(
        (x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(
        buffer_size).batch(batch_size).repeat()
    val_data_single = tf.data.Dataset.from_tensor_slices(
        (x_val_single, y_val_single))
    val_data_single = val_data_single.batch(batch_size).repeat()
    # AutoML: automatically build and fine tune the hybrid model
    def build_att_lstm_model(hp):
        # input layer
        x_input = Input(
            shape=(
                x_train_single.shape[1],
                x_train_single.shape[2]))
        # LSTM layer
        lstm_layer1 = tf.keras.layers.LSTM(
            x_train_single.shape[2],
            return_sequences=True)(x_input)
        drop_layer1 = tf.keras.layers.Dropout(
            hp.Float(
                'Dropout_rate',
                min_value=0,
                max_value=0.9,
                step=0.3))(lstm_layer1)
        lstm_layer2 = tf.keras.layers.LSTM(
            x_train_single.shape[2],
            return_sequences=True)(drop_layer1)
        drop_layer2 = tf.keras.layers.Dropout(
            hp.Float(
                'Dropout_rate',
                min_value=0,
                max_value=0.9,
                step=0.3))(lstm_layer2)
        lstm_layer3 = tf.keras.layers.LSTM(
            x_train_single.shape[2],
            return_sequences=True)(drop_layer2)
        drop_layer3 = tf.keras.layers.Dropout(
            hp.Float(
                'Dropout_rate',
                min_value=0,
                max_value=0.9,
                step=0.3))(lstm_layer3)
        # residual step -> concatenate outputs of layers
        residual_net = tf.keras.layers.Concatenate(
            axis=-1)([x_input, drop_layer1, drop_layer2, drop_layer3])
        # attention layer
        att_layer = attention()(residual_net)
        multiply_val = Multiply()([residual_net, att_layer])
        # other lstm layers
        lstm_layer4 = tf.keras.layers.LSTM(
            4 * x_train_single.shape[2],
            return_sequences=True)(multiply_val)
        drop_s = tf.keras.layers.Dropout(
            hp.Float(
                'Dropout_rate',
                min_value=0,
                max_value=0.9,
                step=0.3))(lstm_layer4)
        # tune number of lstm layers
        for i in range(hp.Int('n_layers', 0, 2)):
            lstm_s = tf.keras.layers.LSTM(
                hp.Int(
                    f'lstm_{
                        i +
                        2}_units',
                    min_value=x_train_single.shape[2],
                    max_value=4 *
                    x_train_single.shape[2],
                    step=x_train_single.shape[2]),
                return_sequences=True)(drop_s)
            drop_s = tf.keras.layers.Dropout(
                hp.Float(
                    'Dropout_rate',
                    min_value=0,
                    max_value=0.9,
                    step=0.3))(lstm_s)
        last_lstm_layer = tf.keras.layers.LSTM(
            hp.Int(
                'last_units',
                min_value=x_train_single.shape[2],
                max_value=4 *
                x_train_single.shape[2],
                step=x_train_single.shape[2]))(drop_s)
        drop_layer5 = tf.keras.layers.Dropout(
            hp.Float(
                'Dropout_rate',
                min_value=0,
                max_value=0.9,
                step=0.3))(last_lstm_layer)
        # final dense layers
        dense_layer_1 = tf.keras.layers.Dense(
            hp.Int(
                'last_units',
                min_value=x_train_single.shape[2],
                max_value=4 *
                x_train_single.shape[2],
                step=x_train_single.shape[2]),
            activation='relu')(drop_layer5)
        dense_layer_2 = tf.keras.layers.Dense(
            hp.Int(
                'last_units',
                min_value=x_train_single.shape[2],
                max_value=4 *
                x_train_single.shape[2],
                step=x_train_single.shape[2]),
            activation='relu')(dense_layer_1)
        # output layer
        out_put = tf.keras.layers.Dense(1, activation='relu')(dense_layer_2)
        # build model
        model = Model(x_input, out_put)
        model.compile(loss='mae', optimizer='adam')
        print(model.summary())
        return model

    # set early stop to avoid overfitting while training
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)
    # build tuner
    tuner = RandomSearch(
        build_att_lstm_model,
        objective=['loss'],
        max_trials=4,
        executions_per_trial=1,
        seed=42,
        directory='./lstm/tuner/' + date_,
        project_name=f'{v_make}_{v_model}')
    # run tuner
    tuner.search(
        x=x_train_single,
        y=y_train_single,
        epochs=epochs_,
        batch_size=batch_size,
        validation_data=(x_val_single, y_val_single),
        callbacks=[stop_early])
    # pick the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    # build the final model with the best hyperparameters
    single_step_model = tuner.hypermodel.build(best_hp)
    single_step_model.compile(optimizer='adam', loss='mae')
    single_step_history = single_step_model.fit(
        train_data_single,
        epochs=epochs_,
        steps_per_epoch=5,
        validation_data=val_data_single,
        validation_steps=5,
        callbacks=[stop_early])
    # plot history of training
    pyplot.plot(single_step_history.history['loss'], label='train')
    pyplot.plot(single_step_history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.xlabel("epochs")
    pyplot.ylabel("MAE")
    pyplot.savefig(f"./LSTM/plots/{date_}/{v_make}_{v_model}.png")
    pyplot.clf()
    # save the model
    single_step_model.save(f"./LSTM/{v_make}_{v_model}_{date_}_model.keras")
    # create output df
    outdf = pd.DataFrame(single_step_model.predict(all_data_x))
    outdf = outdf * max_train_sale
    outdf.loc[:, "date"] = input_df.loc[past_history + 1:].date.values
    outdf.rename(columns={0: "sales"}, inplace=True)
    outdf = pd.concat((input_df[input_df.date < outdf.date.min()][[
                      "date", "sales"]], outdf)).sort_values("date").reset_index(drop=True)
    backward = 7
    steps_forward = forecast_steps
    input_df.date = pd.to_datetime(input_df.date)
    current_day = pd.to_datetime(outdf.date.max())
    temp_out = dataset.copy()
    # create the data point for next month sale prediction
    data_1 = []
    start_index = len(temp_out) - backward
    end_index = len(temp_out)
    indices = range(start_index, end_index, 1)
    data_1.append(temp_out[indices])
    output = []
    output_inv = []
    future_date = []
    # repeat prediction for look-forward months
    for step in range(0, steps_forward):
        # predict
        data_2 = []
        pred = single_step_model.predict(np.array(data_1))[0][0]
        data_2.append(pred)
        data_2.append(temp_out[-1][1])
        output.append(pred)
        output_inv.append(pred * max_train_sale)
        # next date point
        next_date = current_day + relativedelta(months=1)
        future_date.append(next_date)
        for i in range(1, 12):
            x = 1 if next_date.month == i else 0
            x = (x - df_mean[i + 1]) / df_sd[i + 1]
            data_2.append(x)
        data_featuers = temp_out[-1][13:]
        data_2.extend(data_featuers)
        data_3 = data_1.copy()
        data_3[0] = np.delete(data_3[0], 0, axis=0)
        data_3[0] = np.vstack((data_3[0], data_2))
        current_day = next_date
        data_1 = data_3.copy()
    forecast_df = pd.DataFrame(
        zip(future_date, output_inv), columns=["date", "sales"])
    # apply monthly growth factor
    model_df2 = input_df.copy()
    max_date = pd.to_datetime(model_df2.date.max())
    p_sale = model_df2.iloc[-1].sales
    month = max_date.month
    lstm_sale_lst = list(forecast_df.sales)
    model_df2["month"] = model_df2.date.apply(lambda x: x.month)
    model_df2["growth"] = model_df2["sales"] / model_df2["sales"].shift(1)
    sale_min_growth = dict(model_df2.groupby("month").growth.agg(min))
    sale_max_growth = dict(model_df2.groupby("month").growth.agg(max))
    forecast_df2 = []
    # repeat prediction for look-forward months
    for i in range(steps_forward):
        sale = lstm_sale_lst[i]
        p_sale_max = p_sale * sale_max_growth[month]
        p_sale_min = p_sale * sale_min_growth[month]
        month = (month + 1) % 12
        if month == 0:
            month = 12
        else:
            pass
        sale = p_sale_max if max(sale, p_sale_max) != p_sale_max else sale
        sale = p_sale_min if min(sale, p_sale_min) != p_sale_min else sale
        forecast_df2.append(sale)
        p_sale = sale
    forecast_df.sale = forecast_df2
    final_df = pd.concat(
        (outdf, forecast_df)).sort_values("date").reset_index(
        drop=True)
    final_df.sales.clip(lower=0, inplace=True)
    return (final_df)
