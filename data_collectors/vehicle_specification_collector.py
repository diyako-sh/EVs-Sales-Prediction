# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

################################ Imports #################################
from selenium import webdriver
import pandas as pd
import time
import inspect
import os
################################ Functions #################################


def gas_veh_spec(make, model, year, spec_df_final):
    """
    gas_veh_spec extracts specification data for gasoline vehicles

    Parameters
    ----------
    make : str
        make of the vehicle
    model : str
        model of the vehicle
    year : str
        year
    spec_df_final : dataframe
        the dataframe containing vehicles' specifications.

    Returns
    -------
    spec_df_final: dataframe
        the updated input dataframe
    """
    # set driver
    driver = webdriver.Chrome(executable_path=r"PATH_TO_chromedriver.exe")
    driver.get(r"https://www.thecarconnection.com/specifications")
    # define empty lists for specification
    make_lst = []
    model_lst = []
    model_lst = []
    year_lst = []
    style_lst = []
    fuel_tank_capacity = []
    price_lst = []
    mpg_comb_lst = []
    eng_hp_lst = []
    mileage = []
    safety_features1 = []
    safety_features2 = []
    safety_features3 = []
    safety_features4 = []
    safety_features5 = []
    safety_features6 = []
    safety_features7 = []
    safety_features8 = []
    safety_features9 = []
    safety_features10 = []
    safety_features11 = []
    safety_features12 = []
    safety_features13 = []
    safety_features14 = []
    safety_features15 = []
    safety_features16 = []
    safety_features17 = []
    safety_features18 = []
    safety_score = []
    other_features = []
    len_other_features = []
    warranty_features1 = []
    warranty_features2 = []
    max_price = []
    min_price = []
    mean_price = []
    max_mpg = []
    min_mpg = []
    mean_mpg = []
    max_mileage = []
    min_mileage = []
    mean_mileage = []
    max_en_power = []
    min_en_power = []
    mean_en_power = []
    max_safety_score = []
    min_safety_score = []
    mean_safety_score = []
    max_options_score = []
    min_options_score = []
    mean_options_score = []
    max_warranty_mile = []
    min_warranty_mile = []
    mean_warranty_mile = []
    max_warranty_years = []
    min_warranty_years = []
    mean_warranty_years = []
    make_1 = []
    model_1 = []
    year_1 = []
    driver.get(
        r"https://www.thecarconnection.com/specifications/" +
        make +
        "_" +
        model +
        "_" +
        year)
    # trims are the same vehicle with different features and price
    trims_icon = driver.find_element_by_css_selector("#change-trim")
    trims_icon.click()
    time.sleep(5)
    trims_lst = driver.find_elements_by_class_name("name")
    trims_lst[0].click()
    time.sleep(5)
    # extract price
    try:
        pr = driver.find_element_by_css_selector(
            "#left-column > div.specs-top-bar > div > a > span").text
        pr = pr.replace("$", "")
        pr = pr.replace(",", "")
        price_lst.append(int(pr))
    except BaseException:
        price_lst.append(None)
    make_lst.append(make)
    model_lst.append(model)
    year_lst.append(int(year))
    # extract style
    try:
        st = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(1) > div > div > div:nth-child(5) > span.value").text
        style_lst.append(st)
    except BaseException:
        style_lst.append(None)
    # extract fuel tank capacity
    try:
        ftc = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(4) > div > div:nth-child(1) > div.specs-set-item.item-top-border > span.value").text
        fuel_tank_capacity.append(float(ftc))
    except BaseException:
        fuel_tank_capacity.append(None)
    # extract MPG (or its equivalent for EVs)
    try:
        mp1 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(4) > div > div:nth-child(2) > div.specs-set-item.item-top-border > span.value").text
        mpg_comb_lst.append(int(mp1))
    except BaseException:
        try:
            mp1 = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div:nth-child(2) > div.specs-set-item.item-top-border > span.value").text.split(" ")[0]
            mpg_comb_lst.append(int(mp1))
        except BaseException:
            mpg_comb_lst.append(None)
    # extract engine horse power
    try:
        en = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(6) > span.value").text
        eng_hp_lst.append(int(en))
    except BaseException:
        try:
            en = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(6) > span.value").text
            en = en.split("@")[0].replace(" ", "")
            eng_hp_lst.append(int(en))
        except BaseException:
            try:
                en = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(7) > span.value").text
                en = en.split("@")[0].replace(" ", "")
                eng_hp_lst.append(int(en))

            except BaseException:
                eng_hp_lst.append(None)
    # extract mileage
    try:
        mileage.append(round(float(int(mp1) * float(ftc))))
    except BaseException:
        mileage.append(None)
    # safety factors(sf) of vehicels
    sf1 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div.specs-set-item.item-top-border > span.value").text
    if sf1 == "Yes":
        sf1 = 1
    elif sf1 == "No":
        sf1 = 0
    safety_features1.append(sf1)
    sf2 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(3) > span.value").text
    if sf2 == "Yes":
        sf2 = 1
    elif sf2 == "No":
        sf2 = 0
    safety_features2.append(sf2)
    sf3 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(4) > span.value").text
    if sf3 == "Yes":
        sf3 = 1
    elif sf3 == "No":
        sf3 = 0
    safety_features3.append(sf3)
    sf4 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(5) > span.value").text
    if sf4 == "Yes":
        sf4 = 1
    elif sf4 == "No":
        sf4 = 0
    safety_features4.append(sf4)
    sf5 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(6) > span.value").text
    if sf5 == "Yes":
        sf5 = 1
    elif sf5 == "No":
        sf5 = 0
    safety_features5.append(sf5)
    sf6 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(7) > span.value").text
    if sf6 == "Yes":
        sf6 = 1
    elif sf6 == "No":
        sf6 = 0
    safety_features6.append(sf6)
    sf7 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(8) > span.value").text
    if sf7 == "Yes":
        sf7 = 1
    elif sf7 == "No":
        sf7 = 0
    safety_features7.append(sf7)
    sf8 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(9) > span.value").text
    if sf8 == "Yes":
        sf8 = 1
    elif sf8 == "No":
        sf8 = 0
    safety_features8.append(sf8)
    sf9 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(10) > span.value").text
    if sf9 == "Yes":
        sf9 = 1
    elif sf9 == "No":
        sf9 = 0
    safety_features9.append(sf9)
    sf10 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(11) > span.value").text
    if sf10 == "Yes":
        sf10 = 1
    elif sf10 == "No":
        sf10 = 0
    safety_features10.append(sf10)
    sf11 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(12) > span.value").text
    if sf11 == "Yes":
        sf11 = 1
    elif sf11 == "No":
        sf11 = 0
    safety_features11.append(sf11)
    sf12 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(13) > span.value").text
    if sf12 == "Yes":
        sf12 = 1
    elif sf12 == "No":
        sf12 = 0
    safety_features12.append(sf12)
    sf13 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(14) > span.value").text
    if sf13 == "Yes":
        sf13 = 1
    elif sf13 == "No":
        sf13 = 0
    safety_features13.append(sf13)
    sf14 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(15) > span.value").text
    if sf14 == "Yes":
        sf14 = 1
    elif sf14 == "No":
        sf14 = 0
    safety_features14.append(sf14)
    sf15 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(16) > span.value").text
    if sf15 == "Yes":
        sf15 = 1
    elif sf15 == "No":
        sf15 = 0
    safety_features15.append(sf15)

    sf16 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(17) > span.value").text
    if sf16 == "Yes":
        sf16 = 1
    elif sf16 == "No":
        sf16 = 0
    safety_features16.append(sf16)
    sf17 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(18) > span.value").text
    if sf17 == "Yes":
        sf17 = 1
    elif sf17 == "No":
        sf17 = 0
    safety_features17.append(sf17)
    sf18 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(19) > span.value").text
    if sf18 == "Yes":
        sf18 = 1
    elif sf18 == "No":
        sf18 = 0
    safety_features18.append(sf18)
    # calculate safety score
    safety_score.append(
        float(
            (sf1 +
             sf2 +
             sf3 +
             sf4 +
             sf5 +
             sf6 +
             sf7 +
             sf8 +
             sf9 +
             sf10 +
             sf11 +
             sf12 +
             sf13 +
             sf14 +
             sf15 +
             sf16 +
             sf17 +
             sf18) /
            18))
    # extract other options(featues) of vehicles
    of = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(20) > span.value").text.split("\n")
    other_features.extend([of])
    len_other_features.append(len(of))
    # extract warranty features
    wa1 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(7) > div > div > div.specs-set-item.item-top-border > span.value").text
    wa1 = wa1.replace(",", "")
    warranty_features1.append(int(wa1))
    wa2 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(7) > div > div > div:nth-child(3) > span.value").text
    warranty_features2.append(int(wa2))
    # extract information of trims( the same vehicle with different features
    # and price)
    for t in range(len(trims_lst) - 1):
        trims_icon = driver.find_element_by_css_selector("#change-trim")
        trims_icon.click()
        time.sleep(5)
        trims_lst = driver.find_elements_by_class_name("name")
        trims_lst[t + 1].click()
        time.sleep(5)
        # extract price
        try:
            pr = driver.find_element_by_css_selector(
                "#left-column > div.specs-top-bar > div > a > span").text
            pr = pr.replace("$", "")
            pr = pr.replace(",", "")
            price_lst.append(int(pr))
        except BaseException:
            price_lst.append(None)

        make_lst.append(make)
        model_lst.append(model)
        year_lst.append(int(year))
        # extract style
        try:
            st = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(1) > div > div > div:nth-child(5) > span.value").text
            style_lst.append(st)
        except BaseException:
            style_lst.append(None)
        # extract fuel tank capacity
        try:
            ftc = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div:nth-child(1) > div.specs-set-item.item-top-border > span.value").text
            fuel_tank_capacity.append(float(ftc))
        except BaseException:
            fuel_tank_capacity.append(None)
        # extract MPG (or its equivalent for EVs)
        try:
            mp1 = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div:nth-child(2) > div.specs-set-item.item-top-border > span.value").text
            mpg_comb_lst.append(int(mp1))
        except BaseException:
            try:
                mp1 = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(4) > div > div:nth-child(2) > div.specs-set-item.item-top-border > span.value").text.split(" ")[0]
                mpg_comb_lst.append(int(mp1))
            except BaseException:
                mpg_comb_lst.append(None)
        # extract engine horse power
        try:
            en = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(6) > span.value").text
            eng_hp_lst.append(int(en))
        except BaseException:
            try:
                en = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(6) > span.value").text
                en = en.split("@")[0].replace(" ", "")
                eng_hp_lst.append(int(en))
            except BaseException:
                try:
                    en = driver.find_element_by_css_selector(
                        "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(7) > span.value").text
                    en = en.split("@")[0].replace(" ", "")
                    eng_hp_lst.append(int(en))

                except BaseException:
                    eng_hp_lst.append(None)
        # extract mileage
        try:
            mileage.append(round(float(int(mp1) * float(ftc))))
        except BaseException:
            mileage.append(None)
        # safety factors(sf) of vehicels
        sf1 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div.specs-set-item.item-top-border > span.value").text
        if sf1 == "Yes":
            sf1 = 1
        elif sf1 == "No":
            sf1 = 0
        safety_features1.append(sf1)

        sf2 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(3) > span.value").text
        if sf2 == "Yes":
            sf2 = 1
        elif sf2 == "No":
            sf2 = 0
        safety_features2.append(sf2)

        sf3 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(4) > span.value").text
        if sf3 == "Yes":
            sf3 = 1
        elif sf3 == "No":
            sf3 = 0
        safety_features3.append(sf3)
        sf4 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(5) > span.value").text
        if sf4 == "Yes":
            sf4 = 1
        elif sf4 == "No":
            sf4 = 0
        safety_features4.append(sf4)
        sf5 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(6) > span.value").text
        if sf5 == "Yes":
            sf5 = 1
        elif sf5 == "No":
            sf5 = 0
        safety_features5.append(sf5)
        sf6 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(7) > span.value").text
        if sf6 == "Yes":
            sf6 = 1
        elif sf6 == "No":
            sf6 = 0
        safety_features6.append(sf6)
        sf7 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(8) > span.value").text
        if sf7 == "Yes":
            sf7 = 1
        elif sf7 == "No":
            sf7 = 0
        safety_features7.append(sf7)
        sf8 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(9) > span.value").text
        if sf8 == "Yes":
            sf8 = 1
        elif sf8 == "No":
            sf8 = 0
        safety_features8.append(sf8)
        sf9 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(10) > span.value").text
        if sf9 == "Yes":
            sf9 = 1
        elif sf9 == "No":
            sf9 = 0
        safety_features9.append(sf9)
        sf10 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(11) > span.value").text
        if sf10 == "Yes":
            sf10 = 1
        elif sf10 == "No":
            sf10 = 0
        safety_features10.append(sf10)
        sf11 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(12) > span.value").text
        if sf11 == "Yes":
            sf11 = 1
        elif sf11 == "No":
            sf11 = 0
        safety_features11.append(sf11)
        sf12 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(13) > span.value").text
        if sf12 == "Yes":
            sf12 = 1
        elif sf12 == "No":
            sf12 = 0
        safety_features12.append(sf12)
        sf13 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(14) > span.value").text
        if sf13 == "Yes":
            sf13 = 1
        elif sf13 == "No":
            sf13 = 0
        safety_features13.append(sf13)
        sf14 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(15) > span.value").text
        if sf14 == "Yes":
            sf14 = 1
        elif sf14 == "No":
            sf14 = 0
        safety_features14.append(sf14)
        sf15 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(16) > span.value").text
        if sf15 == "Yes":
            sf15 = 1
        elif sf15 == "No":
            sf15 = 0
        safety_features15.append(sf15)
        sf16 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(17) > span.value").text
        if sf16 == "Yes":
            sf16 = 1
        elif sf16 == "No":
            sf16 = 0
        safety_features16.append(sf16)
        sf17 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(18) > span.value").text
        if sf17 == "Yes":
            sf17 = 1
        elif sf17 == "No":
            sf17 = 0
        safety_features17.append(sf17)
        sf18 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(19) > span.value").text
        if sf18 == "Yes":
            sf18 = 1
        elif sf18 == "No":
            sf18 = 0
        safety_features18.append(sf18)
        # calculate safety score
        safety_score.append(
            float(
                (sf1 +
                 sf2 +
                 sf3 +
                 sf4 +
                 sf5 +
                 sf6 +
                 sf7 +
                 sf8 +
                 sf9 +
                 sf10 +
                 sf11 +
                 sf12 +
                 sf13 +
                 sf14 +
                 sf15 +
                 sf16 +
                 sf17 +
                 sf18) /
                18))
        # extract other options(featues) of vehicles
        of = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(20) > span.value").text.split("\n")
        other_features.extend([of])
        len_other_features.append(len(of))
        # extract warranty features
        wa1 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(7) > div > div > div.specs-set-item.item-top-border > span.value").text
        wa1 = wa1.replace(",", "")
        warranty_features1.append(int(wa1))

        wa2 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(7) > div > div > div:nth-child(3) > span.value").text
        warranty_features2.append(int(wa2))
    # summarize collected data into a dataframe
    spec_df1 = pd.DataFrame({"make": make_lst,
                             "model": model_lst,
                             "year": year_lst,
                             "trim": style_lst,
                             "price": price_lst,
                             "mpg_Combined": mpg_comb_lst,
                             "Fuel Tank Capacity_Approx": fuel_tank_capacity,
                             "Mileage": mileage,
                             "en_hp": eng_hp_lst,
                             "Safety_Air Bag-Frontal-Driver": safety_features1,
                             "Safety_Air Bag-Frontal-Passenger": safety_features2,
                             "Safety_Air Bag-Passenger Switch": safety_features3,
                             "Safety_Air Bag-Side Body-Front": safety_features4,
                             "Safety_Air Bag-Side Body-Rear": safety_features5,
                             "Safety_Air Bag-Side Head-Front": safety_features6,
                             "Safety_Air Bag-Side Head-Rear": safety_features7,
                             "Safety_Brakes-ABS": safety_features8,
                             "Safety_Child Safety Rear Door Locks": safety_features9,
                             "Safety_Daytime Running Lights": safety_features10,
                             "Safety_Traction Control": safety_features11,
                             "Safety_Night Vision": safety_features12,
                             "Safety_Rollover Protection Bars": safety_features13,
                             "Safety_Fog Lamps": safety_features14,
                             "Safety_Parking Aid": safety_features15,
                             "Safety_Tire Pressure Monitor": safety_features16,
                             "Safety_Back-Up Camera": safety_features17,
                             "Safety_Stability Control": safety_features18,
                             "Safety_Score": safety_score,
                             "Other Features": other_features,
                             "len_other_features": len_other_features,
                             "Warranty_Miles": warranty_features1,
                             "Warranty_Years": warranty_features2})
    # prepare data (considering three values for continuous data: min, max,
    # and average)
    spec_df1.fillna(method="ffill", inplace=True)
    spec_df1.fillna(method="backfill", inplace=True)
    max_price.append(spec_df1.price.max())
    min_price.append(spec_df1.price.min())
    mean_price.append(round(spec_df1.price.mean()))
    max_mpg.append(spec_df1.mpg_Combined.max())
    min_mpg.append(spec_df1.mpg_Combined.min())
    mean_mpg.append(round(spec_df1.mpg_Combined.mean()))
    max_mileage.append(spec_df1.Mileage.max())
    min_mileage.append(spec_df1.Mileage.min())
    mean_mileage.append(round(spec_df1.Mileage.mean()))
    max_en_power.append(spec_df1.en_hp.max())
    min_en_power.append(spec_df1.en_hp.min())
    mean_en_power.append(round(spec_df1.en_hp.mean()))
    max_safety_score.append(spec_df1.Safety_Score.max())
    min_safety_score.append(spec_df1.Safety_Score.min())
    mean_safety_score.append(spec_df1.Safety_Score.mean())
    max_options_score.append(spec_df1.len_other_features.max())
    min_options_score.append(spec_df1.len_other_features.min())
    mean_options_score.append(spec_df1.len_other_features.mean())
    max_warranty_mile.append(spec_df1.Warranty_Miles.max())
    min_warranty_mile.append(spec_df1.Warranty_Miles.min())
    mean_warranty_mile.append(spec_df1.Warranty_Miles.mean())
    max_warranty_years.append(spec_df1.Warranty_Years.max())
    min_warranty_years.append(spec_df1.Warranty_Years.min())
    mean_warranty_years.append(spec_df1.Warranty_Years.mean())
    make_1.append(make)
    model_1.append(model)
    year_1.append(year)
    # generate final specification dataframe for the vehicle
    spec_df2 = pd.DataFrame({"make": make_1,
                            "model": model_1,
                             "year": year_1,
                             "max_price": max_price,
                             "min_price": min_price,
                             "mean_price": mean_price,
                             "max_mpg": max_mpg,
                             "min_mpg": min_mpg,
                             "mean_mpg": mean_mpg,
                             "max_mileage": max_mileage,
                             "min_mileage": min_mileage,
                             "mean_mileage": mean_mileage,
                             "max_en_power": max_en_power,
                             "min_en_power": min_en_power,
                             "mean_en_power": mean_en_power,
                             "max_safety_score": max_safety_score,
                             "min_safety_score": min_safety_score,
                             "mean_safety_score": mean_safety_score,
                             "max_options_score": max_options_score,
                             "min_options_score": min_options_score,
                             "mean_options_score": mean_options_score,
                             "max_warranty_mile": max_warranty_mile,
                             "min_warranty_mile": min_warranty_mile,
                             "mean_warranty_mile": mean_warranty_mile,
                             "max_warranty_years": max_warranty_years,
                             "min_warranty_years": min_warranty_years,
                             "mean_warranty_years": mean_warranty_years})
    # merge the generated dataframe for a vehicle to the final dataframe
    # containing all vehicles' specification
    spec_df_final = pd.concat((spec_df_final, spec_df2), axis=0)
    return (spec_df_final)


def elc_veh_spec(make, model, year, spec_df_final):
    """
    elc_veh_spec extracts specification data for EVs

    Parameters
    ----------
    make : str
        make of the vehicle
    model : str
        model of the vehicle
    year : str
        year
    spec_df_final : dataframe
        the dataframe containing vehicles' specifications.

    Returns
    -------
    spec_df_final: dataframe
        the updated input dataframe
    """
    # set driver
    driver = webdriver.Chrome(executable_path=r"PATH_TO_chromedriver.exe")
    driver.get(r"https://www.thecarconnection.com/specifications")
    # define empty lists for specification
    make_lst = []
    model_lst = []
    model_lst = []
    year_lst = []
    style_lst = []
    price_lst = []
    mpg_comb_lst = []
    eng_hp_lst = []
    mileage = []
    safety_features1 = []
    safety_features2 = []
    safety_features3 = []
    safety_features4 = []
    safety_features5 = []
    safety_features6 = []
    safety_features7 = []
    safety_features8 = []
    safety_features9 = []
    safety_features10 = []
    safety_features11 = []
    safety_features12 = []
    safety_features13 = []
    safety_features14 = []
    safety_features15 = []
    safety_features16 = []
    safety_features17 = []
    safety_features18 = []
    safety_score = []
    other_features = []
    len_other_features = []
    warranty_features1 = []
    warranty_features2 = []
    max_price = []
    min_price = []
    mean_price = []
    max_mpg = []
    min_mpg = []
    mean_mpg = []
    max_mileage = []
    min_mileage = []
    mean_mileage = []
    max_en_power = []
    min_en_power = []
    mean_en_power = []
    max_safety_score = []
    min_safety_score = []
    mean_safety_score = []
    max_options_score = []
    min_options_score = []
    mean_options_score = []
    max_warranty_mile = []
    min_warranty_mile = []
    mean_warranty_mile = []
    max_warranty_years = []
    min_warranty_years = []
    mean_warranty_years = []
    make_1 = []
    model_1 = []
    year_1 = []
    driver.get(
        r"https://www.thecarconnection.com/specifications/" +
        make +
        "_" +
        model +
        "_" +
        year)
    # trims are the same vehicle with different features and price
    trims_icon = driver.find_element_by_css_selector("#change-trim")
    trims_icon.click()
    time.sleep(5)
    trims_lst = driver.find_elements_by_class_name("name")
    trims_lst[0].click()
    time.sleep(5)
    # extract price
    try:
        pr = driver.find_element_by_css_selector(
            "#left-column > div.specs-top-bar > div > a > span").text
        pr = pr.replace("$", "")
        pr = pr.replace(",", "")
        price_lst.append(int(pr))
    except BaseException:
        price_lst.append(None)
    make_lst.append(make)
    model_lst.append(model)
    year_lst.append(int(year))
    # extract style
    try:
        st = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(1) > div > div > div:nth-child(5) > span.value").text
        style_lst.append(st)
    except BaseException:

        style_lst.append(None)
    # extract engine horse power
    try:
        eng_hp = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(4) > span.value").text
        eng_hp_lst.append(int(eng_hp))
    except BaseException:
        eng_hp_lst.append(None)
    # extract MPG (or its equivalent for EVs)
    if driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(4) > div > div > div.specs-set-item.item-top-border > span.value").text == "NA":
        try:
            mp1 = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div > div:nth-child(3) > span.value").text
            mpg_comb_lst.append(int(mp1))
        except BaseException:
            mpg_comb_lst.append(None)
        # extract mileage
        try:
            mi_el = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div > div:nth-child(6) > span.value").text

            mileage.append(int(mi_el))

        except BaseException:
            mileage.append(None)
    else:
        # extract MPG (or its equivalent for EVs)
        try:
            mp1 = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(5) > span.value").text
            mpg_comb_lst.append(int(mp1))
        except BaseException:
            mpg_comb_lst.append(None)
        # extract mileage
        try:
            mi_el = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div > div:nth-child(3) > span.value").text
            mileage.append(int(mi_el))
        except BaseException:
            mileage.append(None)
    # safety factors(sf) of vehicels
    sf1 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div.specs-set-item.item-top-border > span.value").text
    if sf1 == "Yes":
        sf1 = 1
    elif sf1 == "No":
        sf1 = 0
    safety_features1.append(sf1)
    sf2 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(3) > span.value").text
    if sf2 == "Yes":
        sf2 = 1
    elif sf2 == "No":
        sf2 = 0
    safety_features2.append(sf2)
    sf3 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(4) > span.value").text
    if sf3 == "Yes":
        sf3 = 1
    elif sf3 == "No":
        sf3 = 0
    safety_features3.append(sf3)
    sf4 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(5) > span.value").text
    if sf4 == "Yes":
        sf4 = 1
    elif sf4 == "No":
        sf4 = 0
    safety_features4.append(sf4)
    sf5 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(6) > span.value").text
    if sf5 == "Yes":
        sf5 = 1
    elif sf5 == "No":
        sf5 = 0
    safety_features5.append(sf5)
    sf6 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(7) > span.value").text
    if sf6 == "Yes":
        sf6 = 1
    elif sf6 == "No":
        sf6 = 0
    safety_features6.append(sf6)
    sf7 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(8) > span.value").text
    if sf7 == "Yes":
        sf7 = 1
    elif sf7 == "No":
        sf7 = 0
    safety_features7.append(sf7)
    sf8 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(9) > span.value").text
    if sf8 == "Yes":
        sf8 = 1
    elif sf8 == "No":
        sf8 = 0
    safety_features8.append(sf8)
    sf9 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(10) > span.value").text
    if sf9 == "Yes":
        sf9 = 1
    elif sf9 == "No":
        sf9 = 0
    safety_features9.append(sf9)
    sf10 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(11) > span.value").text
    if sf10 == "Yes":
        sf10 = 1
    elif sf10 == "No":
        sf10 = 0
    safety_features10.append(sf10)
    sf11 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(12) > span.value").text
    if sf11 == "Yes":
        sf11 = 1
    elif sf11 == "No":
        sf11 = 0
    safety_features11.append(sf11)
    sf12 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(13) > span.value").text
    if sf12 == "Yes":
        sf12 = 1
    elif sf12 == "No":
        sf12 = 0
    safety_features12.append(sf12)
    sf13 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(14) > span.value").text
    if sf13 == "Yes":
        sf13 = 1
    elif sf13 == "No":
        sf13 = 0
    safety_features13.append(sf13)
    sf14 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(15) > span.value").text
    if sf14 == "Yes":
        sf14 = 1
    elif sf14 == "No":
        sf14 = 0
    safety_features14.append(sf14)
    sf15 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(16) > span.value").text
    if sf15 == "Yes":
        sf15 = 1
    elif sf15 == "No":
        sf15 = 0
    safety_features15.append(sf15)
    sf16 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(17) > span.value").text
    if sf16 == "Yes":
        sf16 = 1
    elif sf16 == "No":
        sf16 = 0
    safety_features16.append(sf16)
    sf17 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(18) > span.value").text
    if sf17 == "Yes":
        sf17 = 1
    elif sf17 == "No":
        sf17 = 0
    safety_features17.append(sf17)
    sf18 = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(19) > span.value").text
    if sf18 == "Yes":
        sf18 = 1
    elif sf18 == "No":
        sf18 = 0
    safety_features18.append(sf18)
    # calculate safety score
    safety_score.append(
        float(
            (sf1 +
             sf2 +
             sf3 +
             sf4 +
             sf5 +
             sf6 +
             sf7 +
             sf8 +
             sf9 +
             sf10 +
             sf11 +
             sf12 +
             sf13 +
             sf14 +
             sf15 +
             sf16 +
             sf17 +
             sf18) /
            18))
    # extract other options(featues) of vehicles
    of = driver.find_element_by_css_selector(
        "#specs-categories > div:nth-child(6) > div > div > div:nth-child(20) > span.value").text.split("\n")
    other_features.extend([of])
    len_other_features.append(len(of))
    # extract warranty features
    try:
        wa1 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(7) > div > div > div.specs-set-item.item-top-border > span.value").text
        wa1 = wa1.replace(",", "")
        warranty_features1.append(int(wa1))
    except BaseException:
        warranty_features1.append(None)
    try:
        wa2 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(7) > div > div > div:nth-child(3) > span.value").text
        warranty_features2.append(int(wa2))
    except BaseException:
        warranty_features2.append(None)
    # extract information of trims( the same vehicle with different features
    # and price)
    for t in range(len(trims_lst) - 1):
        trims_icon = driver.find_element_by_css_selector("#change-trim")
        trims_icon.click()
        time.sleep(5)
        trims_lst = driver.find_elements_by_class_name("name")
        trims_lst[t + 1].click()
        time.sleep(5)
        # extract price
        try:
            pr = driver.find_element_by_css_selector(
                "#left-column > div.specs-top-bar > div > a > span").text
            pr = pr.replace("$", "")
            pr = pr.replace(",", "")
            price_lst.append(int(pr))
        except BaseException:
            price_lst.append(None)

        make_lst.append(make)
        model_lst.append(model)
        year_lst.append(int(year))
        # extract style
        try:
            st = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(1) > div > div > div:nth-child(5) > span.value").text
            style_lst.append(st)
        except BaseException:
            style_lst.append(None)
        # extract engine horse power
        try:
            eng_hp = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(4) > span.value").text
            eng_hp_lst.append(int(eng_hp))
        except BaseException:
            eng_hp_lst.append(None)

        if driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(4) > div > div > div.specs-set-item.item-top-border > span.value").text == "NA":
           # extract MPG (or its equivalent for EVs)
            try:
                mp1 = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(4) > div > div > div:nth-child(3) > span.value").text
                mpg_comb_lst.append(int(mp1))
            except BaseException:
                mpg_comb_lst.append(None)
            # extract mileage
            try:
                mi_el = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(4) > div > div > div:nth-child(6) > span.value").text
                mileage.append(int(mi_el))
            except BaseException:
                mileage.append(None)
        else:
            # extract MPG (or its equivalent for EVs)
            try:
                mp1 = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(5) > div > div:nth-child(1) > div:nth-child(5) > span.value").text
                mpg_comb_lst.append(int(mp1))
            except BaseException:
                mpg_comb_lst.append(None)
            # extract mileage
            try:
                mi_el = driver.find_element_by_css_selector(
                    "#specs-categories > div:nth-child(4) > div > div > div:nth-child(3) > span.value").text
                mileage.append(int(mi_el))
            except BaseException:
                mileage.append(None)
        # safety factors(sf) of vehicels
        sf1 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div.specs-set-item.item-top-border > span.value").text
        if sf1 == "Yes":
            sf1 = 1
        elif sf1 == "No":
            sf1 = 0
        safety_features1.append(sf1)
        sf2 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(3) > span.value").text
        if sf2 == "Yes":
            sf2 = 1
        elif sf2 == "No":
            sf2 = 0
        safety_features2.append(sf2)
        sf3 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(4) > span.value").text
        if sf3 == "Yes":
            sf3 = 1
        elif sf3 == "No":
            sf3 = 0
        safety_features3.append(sf3)
        sf4 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(5) > span.value").text
        if sf4 == "Yes":
            sf4 = 1
        elif sf4 == "No":
            sf4 = 0
        safety_features4.append(sf4)
        sf5 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(6) > span.value").text
        if sf5 == "Yes":
            sf5 = 1
        elif sf5 == "No":
            sf5 = 0
        safety_features5.append(sf5)
        sf6 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(7) > span.value").text
        if sf6 == "Yes":
            sf6 = 1
        elif sf6 == "No":
            sf6 = 0
        safety_features6.append(sf6)
        sf7 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(8) > span.value").text
        if sf7 == "Yes":
            sf7 = 1
        elif sf7 == "No":
            sf7 = 0
        safety_features7.append(sf7)
        sf8 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(9) > span.value").text
        if sf8 == "Yes":
            sf8 = 1
        elif sf8 == "No":
            sf8 = 0
        safety_features8.append(sf8)
        sf9 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(10) > span.value").text
        if sf9 == "Yes":
            sf9 = 1
        elif sf9 == "No":
            sf9 = 0
        safety_features9.append(sf9)
        sf10 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(11) > span.value").text
        if sf10 == "Yes":
            sf10 = 1
        elif sf10 == "No":
            sf10 = 0
        safety_features10.append(sf10)
        sf11 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(12) > span.value").text
        if sf11 == "Yes":
            sf11 = 1
        elif sf11 == "No":
            sf11 = 0
        safety_features11.append(sf11)
        sf12 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(13) > span.value").text
        if sf12 == "Yes":
            sf12 = 1
        elif sf12 == "No":
            sf12 = 0
        safety_features12.append(sf12)
        sf13 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(14) > span.value").text
        if sf13 == "Yes":
            sf13 = 1
        elif sf13 == "No":
            sf13 = 0
        safety_features13.append(sf13)
        sf14 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(15) > span.value").text
        if sf14 == "Yes":
            sf14 = 1
        elif sf14 == "No":
            sf14 = 0
        safety_features14.append(sf14)
        sf15 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(16) > span.value").text
        if sf15 == "Yes":
            sf15 = 1
        elif sf15 == "No":
            sf15 = 0
        safety_features15.append(sf15)
        sf16 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(17) > span.value").text
        if sf16 == "Yes":
            sf16 = 1
        elif sf16 == "No":
            sf16 = 0
        safety_features16.append(sf16)
        sf17 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(18) > span.value").text
        if sf17 == "Yes":
            sf17 = 1
        elif sf17 == "No":
            sf17 = 0
        safety_features17.append(sf17)
        sf18 = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(19) > span.value").text
        if sf18 == "Yes":
            sf18 = 1
        elif sf18 == "No":
            sf18 = 0
        safety_features18.append(sf18)
        # calculate safety score
        safety_score.append(
            float(
                (sf1 +
                 sf2 +
                 sf3 +
                 sf4 +
                 sf5 +
                 sf6 +
                 sf7 +
                 sf8 +
                 sf9 +
                 sf10 +
                 sf11 +
                 sf12 +
                 sf13 +
                 sf14 +
                 sf15 +
                 sf16 +
                 sf17 +
                 sf18) /
                18))
        # extract other options(featues) of vehicles
        of = driver.find_element_by_css_selector(
            "#specs-categories > div:nth-child(6) > div > div > div:nth-child(20) > span.value").text.split("\n")
        other_features.extend([of])
        len_other_features.append(len(of))
        # extract warranty features
        try:
            wa1 = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(7) > div > div > div.specs-set-item.item-top-border > span.value").text
            wa1 = wa1.replace(",", "")
            warranty_features1.append(int(wa1))
        except BaseException:
            warranty_features1.append(None)
        try:
            wa2 = driver.find_element_by_css_selector(
                "#specs-categories > div:nth-child(7) > div > div > div:nth-child(3) > span.value").text
            warranty_features2.append(int(wa2))
        except BaseException:
            warranty_features2.append(None)
    time.sleep(5)
    # summarize collected data into a dataframe
    spec_df1 = pd.DataFrame({"make": make_lst,
                             "model": model_lst,
                             "year": year_lst,
                             "trim": style_lst,
                             "price": price_lst,
                             "mpg_Combined": mpg_comb_lst,
                             "Mileage": mileage,
                             "en_hp": eng_hp_lst,
                             "Safety_Air Bag-Frontal-Driver": safety_features1,
                             "Safety_Air Bag-Frontal-Passenger": safety_features2,
                             "Safety_Air Bag-Passenger Switch (On/Off)": safety_features3,
                             "Safety_Air Bag-Side Body-Front": safety_features4,
                             "Safety_Air Bag-Side Body-Rear": safety_features5,
                             "Safety_Air Bag-Side Head-Front": safety_features6,
                             "Safety_Air Bag-Side Head-Rear": safety_features7,
                             "Safety_Brakes-ABS": safety_features8,
                             "Safety_Child Safety Rear Door Locks": safety_features9,
                             "Safety_Daytime Running Lights": safety_features10,
                             "Safety_Traction Control": safety_features11,
                             "Safety_Night Vision": safety_features12,
                             "Safety_Rollover Protection Bars": safety_features13,
                             "Safety_Fog Lamps": safety_features14,
                             "Safety_Parking Aid": safety_features15,
                             "Safety_Tire Pressure Monitor": safety_features16,
                             "Safety_Back-Up Camera": safety_features17,
                             "Safety_Stability Control": safety_features18,
                             "Safety_Score": safety_score,
                             "Other Features": other_features,
                             "len_other_features": len_other_features,
                             "Warranty_Miles": warranty_features1,
                             "Warranty_Years": warranty_features2})
    # prepare data (considering three values for continuous data: min, max,
    # and average)
    spec_df1.fillna(method="ffill", inplace=True)
    spec_df1.fillna(method="backfill", inplace=True)
    try:
        spec_df1 = spec_df1.fillna(0)
    except BaseException:
        pass
    max_price.append(spec_df1.price.max())
    min_price.append(spec_df1.price.min())
    mean_price.append(round(spec_df1.price.mean()))
    max_mpg.append(spec_df1.mpg_Combined.max())
    min_mpg.append(spec_df1.mpg_Combined.min())
    mean_mpg.append(round(spec_df1.mpg_Combined.mean()))
    max_mileage.append(spec_df1.Mileage.max())
    min_mileage.append(spec_df1.Mileage.min())
    mean_mileage.append(round(spec_df1.Mileage.mean()))
    max_en_power.append(spec_df1.en_hp.max())
    min_en_power.append(spec_df1.en_hp.min())
    mean_en_power.append(round(spec_df1.en_hp.mean()))
    max_safety_score.append(spec_df1.Safety_Score.max())
    min_safety_score.append(spec_df1.Safety_Score.min())
    mean_safety_score.append(spec_df1.Safety_Score.mean())
    max_options_score.append(spec_df1.len_other_features.max())
    min_options_score.append(spec_df1.len_other_features.min())
    mean_options_score.append(spec_df1.len_other_features.mean())
    max_warranty_mile.append(spec_df1.Warranty_Miles.max())
    min_warranty_mile.append(spec_df1.Warranty_Miles.min())
    mean_warranty_mile.append(spec_df1.Warranty_Miles.mean())
    max_warranty_years.append(spec_df1.Warranty_Years.max())
    min_warranty_years.append(spec_df1.Warranty_Years.min())
    mean_warranty_years.append(spec_df1.Warranty_Years.mean())
    make_1.append(make)
    model_1.append(model)
    year_1.append(year)
    # generate final specification dataframe for the vehicle
    spec_df2 = pd.DataFrame({"make": make_1,
                            "model": model_1,
                             "year": year_1,
                             "max_price": max_price,
                             "min_price": min_price,
                             "mean_price": mean_price,
                             "max_mpg": max_mpg,
                             "min_mpg": min_mpg,
                             "mean_mpg": mean_mpg,
                             "max_mileage": max_mileage,
                             "min_mileage": min_mileage,
                             "mean_mileage": mean_mileage,
                             "max_en_power": max_en_power,
                             "min_en_power": min_en_power,
                             "mean_en_power": mean_en_power,
                             "max_safety_score": max_safety_score,
                             "min_safety_score": min_safety_score,
                             "mean_safety_score": mean_safety_score,
                             "max_options_score": max_options_score,
                             "min_options_score": min_options_score,
                             "mean_options_score": mean_options_score,
                             "max_warranty_mile": max_warranty_mile,
                             "min_warranty_mile": min_warranty_mile,
                             "mean_warranty_mile": mean_warranty_mile,
                             "max_warranty_years": max_warranty_years,
                             "min_warranty_years": min_warranty_years,
                             "mean_warranty_years": mean_warranty_years})
    # merge the generated dataframe for a vehicle to the final dataframe
    # containing all vehicles' specification
    spec_df_final = pd.concat((spec_df_final, spec_df2), axis=0)
    return (spec_df_final)


################################ Run #################################
if __name__ == "__main__":
    # set workspace
    workspace = os.path.dirname(
        os.path.abspath(
            inspect.getfile(
                inspect.currentframe())))
    # create a directory
    try:
        os.mkdir(workspace + "/spc_vehicle")
    except FileExistsError:
        pass
    # load the initial df
    df = pd.read_csv("./data/init_info_df.csv", index_col=0)
    df = df.drop_duplicates(subset=['make', 'model'])
    # set driver
    driver = webdriver.Chrome(executable_path=r"PATH_TO_chromedriver.exe")
    driver.get(r"https://www.thecarconnection.com/specifications")
    # define lists
    elc_list = []
    gas_list = []
    missed_list = []
    # run for all vehicles
    for m in range(len(df)):
        # pick init data
        make = df.make2[m]
        model = df.model2[m]
        make2 = df.make[m]
        model2 = df.model[m]
        # run for all years
        spec_df_final = pd.DataFrame()
        for y in range(2014, 2021):
            year = str(y)
            driver.get(
                r"https://www.thecarconnection.com/specifications/" +
                make +
                "_" +
                model +
                "_" +
                year)
            try:
                if driver.find_element_by_css_selector(
                        "#specs-categories > div:nth-child(1) > div > div > div.specs-set-item.item-top-border > span.value").text == "Electric":
                    # for EVs
                    elc_list.append(make + "_" + model + "_" + year)
                    spec_df_final = elc_veh_spec(
                        make, model, year, spec_df_final).copy()
                    print(make + "_" + model + "_" + year + " is EV!")
                else:
                    # for other
                    spec_df_final = gas_veh_spec(
                        make, model, year, spec_df_final).copy()
                    gas_list.append(make + "_" + model + "_" + year)
            except BaseException:
                missed_list.append(make + "_" + model + "_" + year)
                print(make + "_" + model + "_" + year + " is not found!")
        spec_df_final.to_csv("./spc_vehicle/" + make2 + "_" + model2 + ".csv")
