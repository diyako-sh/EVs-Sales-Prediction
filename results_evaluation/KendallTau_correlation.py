# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
import pandas as pd
from dateutil.relativedelta import *
import scipy.stats as stats
import numpy as np
from math import *
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 100))
plt.style.use('dark_background')
#################################### Run ####################################
# .
# _______________ Stage1: check the first month's prediction _____________###
data1 = pd.read_csv(
    "./LSTM/result/cross_val/df_validation_first_pred.csv",
    index_col=0)
make = data1.drop_duplicates(["make_model"]).make_model.reset_index(drop=True)
df_type = pd.read_csv("./data/vehicles_type_df.csv", index_col=0)
df_type["make_model"] = df_type["make"] + "_" + df_type["model"]
df_all = data1.merge(df_type, on='make_model', how='left')
month = data1.drop_duplicates(["date"]).date.reset_index(drop=True)
segments = df_all.drop_duplicates(["segment"]).segment.reset_index(drop=True)
e_df_final = pd.DataFrame()
for j in range(len(segments)):
    # pick a segment
    seg_df = df_all[df_all.segment == segments[j]]
    mnth = []
    e_pred = []
    e_real = []
    segs = []
    # calculate shares in each month
    for i in range(len(month)):
        segs.append(segments[j])
        df = seg_df[seg_df.date == month[i]]
        # the predicted share
        df["pred_share"] = df.sales_predict / sum(df.sales_predict)
        # the real  share
        df["real_share"] = df.real_sale / sum(df.real_sale)
        mnth.append(month[i])
        e_df = df[df["type"] != "g"]
        e_share_pred = e_df["pred_share"].sum()
        e_pred.append(e_share_pred)
        e_share_real = e_df["real_share"].sum()
        e_real.append(e_share_real)
    # summarize the results in a dataframe
    e_df = pd.DataFrame({"mnth": mnth, "segments": segs,
                        "e_pred": e_pred, "e_real": e_real})
    if e_df.e_real.sum() > 0:
        # plot the results
        e_df_final = pd.concat([e_df_final, e_df], axis=0)
        e_df.index = e_df.mnth
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        num_ = len(e_df)
        ind = np.arange(num_) + 1
        width = 0.4                    #
        rects1 = ax.bar(ind - width, e_df.e_pred, width, color="lime")
        rects2 = ax.bar(ind, e_df.e_real, width, color="magenta")
        ax.set_ylabel("shares")
        ax.set_xlabel("date")
        ax.set_title(f"{segments[j]}")
        plt.xticks(ind, mnth, rotation=90)
        ax.legend((rects1[0], rects2[0]), ('pred_shares', 'real_shares'))
        s = segments[j].replace("/", "_")
        plt.savefig(f"./LSTM/result/shares/first_pred/{s}_shares.png")
        plt.clf()
    else:
        pass
# calculate the KendallTau score
seg_rnk = []
month_rnk = []
tau_val = []
seg_elec = pd.DataFrame()
for z in range(len(segments)):
    seg_df = df_all[df_all.segment == segments[z]]
    if len(seg_df[seg_df.type != 'g']) > 1:
        seg_elec = pd.concat([seg_elec, seg_df], axis=0)
        for m in range(len(month)):
            seg_rnk.append(segments[z])
            month_rnk.append(month[m])
            df = seg_df[seg_df.date == month[m]]
            df = df.sort_values(
                'sales_predict',
                ascending=False).reset_index(
                drop=True)
            df["pred_rank"] = df.index + 1
            df = df.sort_values(
                'real_sale',
                ascending=False).reset_index(
                drop=True)
            df["real_rank"] = df.index + 1
            tau, p_value = stats.kendalltau(df.real_rank, df.pred_rank)
            tau_val.append(tau)
    else:
        pass
# save final results for the first month's prediction
e_df_final["error"] = abs(e_df_final.e_pred - e_df_final.e_real)
e_df_final.to_csv("./LSTM/result/shares/df_shares_first_pred.csv")
tau_df_1 = pd.DataFrame(
    {"segment": seg_rnk, "month": month_rnk, "tau_val": tau_val})
tau_df_1.to_csv("./LSTM/result/shares/tau_df_first_pred.csv")
seg_elec.to_csv("./LSTM/result/shares/df_elec_segs_first_pred.csv")
# .
# ______________ Stage2: check the scond month's prediction ______________###
data2 = pd.read_csv(
    "./LSTM/result/cross_val/df_validation_second_pred.csv",
    index_col=0)
make = data2.drop_duplicates(["make_model"]).make_model.reset_index(drop=True)
df_all = data2.merge(df_type, on='make_model', how='left')
month = data2.drop_duplicates(["date"]).date.reset_index(drop=True)
segments = df_all.drop_duplicates(["segment"]).segment.reset_index(drop=True)
e_df_final2 = pd.DataFrame()
for j in range(len(segments)):
    # pick a segment
    seg_df = df_all[df_all.segment == segments[j]]
    mnth = []
    e_pred = []
    e_real = []
    segs = []
    # calculate shares in each month
    for i in range(len(month)):
        segs.append(segments[j])
        df = seg_df[seg_df.date == month[i]]
        # the predicted share
        df["pred_share"] = df.sales_predict / sum(df.sales_predict)
        # the real  share
        df["real_share"] = df.real_sale / sum(df.real_sale)
        mnth.append(month[i])
        e_df = df[df["type"] != "g"]
        e_share_pred = e_df["pred_share"].sum()
        e_pred.append(e_share_pred)
        e_share_real = e_df["real_share"].sum()
        e_real.append(e_share_real)
    # summarize the results in a dataframe
    e_df = pd.DataFrame({"mnth": mnth, "segments": segs,
                        "e_pred": e_pred, "e_real": e_real})
    if e_df.e_real.sum() > 0:
        # plot the results
        e_df_final2 = pd.concat([e_df_final2, e_df], axis=0)
        e_df.index = e_df.mnth
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        num_ = len(e_df)
        ind = np.arange(num_) + 1
        width = 0.4                    #
        rects1 = ax.bar(ind - width, e_df.e_pred, width, color="lime")
        rects2 = ax.bar(ind, e_df.e_real, width, color="magenta")
        ax.set_ylabel("shares")
        ax.set_xlabel("date")
        ax.set_title(f"{segments[j]}")
        plt.xticks(ind, mnth, rotation=90)
        ax.legend((rects1[0], rects2[0]), ('pred_shares', 'real_shares'))
        s = segments[j].replace("/", "_")
        plt.savefig(f"./LSTM/result/shares/second_pred/{s}_shares.png")
        plt.clf()
    else:
        pass
# calculate the KendallTau score
seg_rnk = []
month_rnk = []
tau_val = []
seg_elec2 = pd.DataFrame()
for z in range(len(segments)):
    seg_df = df_all[df_all.segment == segments[z]]
    if len(seg_df[seg_df.type != 'g']) > 1:
        seg_elec2 = pd.concat([seg_elec2, seg_df], axis=0)
        for m in range(len(month)):
            seg_rnk.append(segments[z])
            month_rnk.append(month[m])
            df = seg_df[seg_df.date == month[m]]
            df = df.sort_values(
                'sales_predict',
                ascending=False).reset_index(
                drop=True)
            df["pred_rank"] = df.index + 1
            df = df.sort_values(
                'real_sale',
                ascending=False).reset_index(
                drop=True)
            df["real_rank"] = df.index + 1
            tau, p_value = stats.kendalltau(df.real_rank, df.pred_rank)
            tau_val.append(tau)
    else:
        pass
# save final results for the second month's prediction
e_df_final2["error"] = abs(e_df_final2.e_pred - e_df_final2.e_real)
e_df_final2.to_csv("./LSTM/result/shares/df_shares_secpnd_pred.csv")
tau_df2 = pd.DataFrame(
    {"segment": seg_rnk, "month": month_rnk, "tau_val": tau_val})
tau_df2.to_csv("./LSTM/result/shares/tau_df_second_pred.csv")
seg_elec2.to_csv("./LSTM/result/shares/df_elec_segs_second_pred.csv")
# .
# ______________ Stage3: check the third month's prediction ______________###
data3 = pd.read_csv(
    "./LSTM/result/cross_val/df_validation_third_pred.csv",
    index_col=0)
make = data3.drop_duplicates(["make_model"]).make_model.reset_index(drop=True)
df_all = data3.merge(df_type, on='make_model', how='left')
month = data3.drop_duplicates(["date"]).date.reset_index(drop=True)
e_df_final3 = pd.DataFrame()
for j in range(len(segments)):
    # pick a segment
    seg_df = df_all[df_all.segment == segments[j]]
    mnth = []
    e_pred = []
    e_real = []
    segs = []
    # calculate shares in each month
    for i in range(len(month)):
        segs.append(segments[j])
        df = seg_df[seg_df.date == month[i]]
        # the predicted share
        df["pred_share"] = df.sales_predict / sum(df.sales_predict)
        # the real  share
        df["real_share"] = df.real_sale / sum(df.real_sale)
        mnth.append(month[i])
        e_df = df[df["type"] != "g"]
        e_share_pred = e_df["pred_share"].sum()
        e_pred.append(e_share_pred)
        e_share_real = e_df["real_share"].sum()
        e_real.append(e_share_real)
    # summarize the results in a dataframe
    e_df = pd.DataFrame({"mnth": mnth, "segments": segs,
                        "e_pred": e_pred, "e_real": e_real})
    if e_df.e_real.sum() > 0:
        # plot the results
        e_df_final3 = pd.concat([e_df_final3, e_df], axis=0)
        e_df.index = e_df.mnth
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        num_ = len(e_df)
        ind = np.arange(num_) + 1
        width = 0.4                    #
        rects1 = ax.bar(ind - width, e_df.e_pred, width, color="lime")
        rects2 = ax.bar(ind, e_df.e_real, width, color="magenta")
        ax.set_ylabel("shares")
        ax.set_xlabel("date")
        ax.set_title(f"{segments[j]}")
        plt.xticks(ind, mnth, rotation=90)
        ax.legend((rects1[0], rects2[0]), ('pred_shares', 'real_shares'))
        s = segments[j].replace("/", "_")
        plt.savefig(f"./LSTM/result/shares/third_pred/{s}_shares.png")
        plt.clf()
    else:
        pass
# calculate the KendallTau score
seg_rnk = []
month_rnk = []
tau_val = []
seg_elec3 = pd.DataFrame()
for z in range(len(segments)):
    seg_df = df_all[df_all.segment == segments[z]]
    if len(seg_df[seg_df.type != 'g']) > 1:
        seg_elec3 = pd.concat([seg_elec3, seg_df], axis=0)
        for m in range(len(month)):
            seg_rnk.append(segments[z])
            month_rnk.append(month[m])
            df = seg_df[seg_df.date == month[m]]
            df = df.sort_values(
                'sales_predict',
                ascending=False).reset_index(
                drop=True)
            df["pred_rank"] = df.index + 1
            df = df.sort_values(
                'real_sale',
                ascending=False).reset_index(
                drop=True)
            df["real_rank"] = df.index + 1
            tau, p_value = stats.kendalltau(df.real_rank, df.pred_rank)
            tau_val.append(tau)
    else:
        pass
# save final results for the second month's prediction
e_df_final3["error"] = abs(e_df_final3.e_pred - e_df_final3.e_real)
e_df_final3.to_csv("./LSTM/result/shares/df_shares_third_pred.csv")
tau_df3 = pd.DataFrame(
    {"segment": seg_rnk, "month": month_rnk, "tau_val": tau_val})
tau_df3.to_csv("./LSTM/result/shares/tau_df_third_pred.csv")
seg_elec3.to_csv("./LSTM/result/shares/df_elec_segs_third_pred.csv")
