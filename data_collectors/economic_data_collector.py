# -*- coding: utf-8 -*-
"""
@author: Diyako
"""

############################# Imports ###############################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
################################ Run #################################
# set start and end date
sdt = dt.datetime(2014, 1, 1)
edt = dt.datetime(2020, 9, 2)
# set the Federal Reserve Economic Data(FRED) API
fred = Fred(api_key='YOUR_FRED_API')
# ___________________________________________
# Gross Domestic Product_Frequency:Quarterly
data1 = pd.DataFrame(fred.get_series('GDP', sdt, edt), columns=['GDP'])
data1.to_csv("GDP_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data1.index, data1['GDP'])
plt.title('Gross Domestic Product')
plt.xlabel('date')
plt.ylabel('Billions of Dollars')
plt.savefig("GDP_index")
# _______________________________________________________________
# Consumer Price Index for All Urban Consumers_Frequency:Monthly
data2 = pd.DataFrame(fred.get_series('CPIAUCSL', sdt, edt), columns=['CPI'])
data2.to_csv("CPI_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data2.index, data2['CPI'])
plt.title('Consumer Price Index for All Urban Consumers')
plt.xlabel('date')
plt.ylabel('Index 1982-1984=100')
plt.savefig("CPI_index")
# ____________________________________________________________
# Producer Price Index by Industry: Transportation Industries
data3 = pd.DataFrame(
    fred.get_series(
        'PCUATRANSATRANS',
        sdt,
        edt),
    columns=['PPI'])
data3.to_csv("PPI_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data3.index, data3['PPI'])
plt.title('Producer Price Index by Industry: Transportation Industries')
plt.xlabel('date')
plt.ylabel('Index Dec 2006=100')
plt.savefig("PPI_index")
# ___________________________
# Personal income per capita
data4 = pd.DataFrame(
    fred.get_series(
        'A792RC0Q052SBEA',
        sdt,
        edt),
    columns=['Personal_income_per_capita'])
data4.to_csv("Personal_income_per_capita_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data4.index, data4["Personal_income_per_capita"])
plt.title('Personal income per capita')
plt.xlabel('date')
plt.ylabel('Dollars')
plt.savefig("Personal_income_per_capita_index")
# ___________________________________________________________________
# Finance Rate on Consumer Installment Loans at Commercial Banks, New
# Autos 48 Month Loan
data5 = pd.DataFrame(
    fred.get_series(
        'TERMCBAUTO48NS',
        sdt,
        edt),
    columns=['Finance_Rate_Loan48m'])
data5.to_csv("Finance_Rate_Loan48m_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.scatter(data5.index, data5["Finance_Rate_Loan48m"])
plt.title('Finance Rate on Consumer Installment Loans at Commercial Banks, New Autos 48 Month Loan')
plt.xlabel('Date')
plt.ylabel('Percent')
plt.savefig("Finance_Rate_Loan48m_index")
# ____________________________________________________________________
# Finance Rate on Consumer Installment Loans at Commercial Banks, New
# Autos 60 Month Loan
data6 = pd.DataFrame(
    fred.get_series(
        'RIFLPBCIANM60NM',
        sdt,
        edt),
    columns=['Finance_Rate_Loan60m'])
data6.to_csv("Finance_Rate_Loan60m_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.scatter(data6.index, data6["Finance_Rate_Loan60m"])
plt.title('Finance Rate on Consumer Installment Loans at Commercial Banks, New Autos 60 Month Loan')
plt.xlabel('Date')
plt.ylabel('Percent')
plt.savefig("Finance_Rate_Loan60m_index")
# _______________________________________________________________________
# Consumer Price Index for All Urban Consumers:New Vehicles in U.S. City
# Average _ monthly
data7 = pd.DataFrame(
    fred.get_series(
        'CUUR0000SETA01',
        sdt,
        edt),
    columns=['CPI_NewVehicle'])
data7.to_csv("CPI_NewVehicles_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data7.index, data7['CPI_NewVehicle'])
plt.title(' Consumer Price Index for All Urban Consumers: New Vehicles in U.S. City Average _ monthly')
plt.xlabel('Date')
plt.ylabel('Index 1982-1984=100')
plt.savefig("CPI_NewVehicles_index")
# _______
# S&P500
data8 = pd.DataFrame(fred.get_series('SP500', sdt, edt), columns=['S&P500'])
data8.to_csv("S&P500_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data8.index, data8["S&P500"])
plt.title(' S&P500')
plt.xlabel('Date')
plt.ylabel('Index')
plt.savefig("S&P500_index")
# ____________________________
# Dow Jones Industrial Average
data9 = pd.DataFrame(fred.get_series('DJIA', sdt, edt), columns=['DowJones'])
data9.to_csv("DowJones_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data9.index, data9["DowJones"])
plt.title(' Dow Jones Industrial Average')
plt.xlabel('Date')
plt.ylabel('Index')
plt.savefig("DowJones_index")
# _______________________________
# Consumer confidence index (CCI)
data10 = pd.DataFrame(
    fred.get_series(
        'CSCICP03USM665S',
        sdt,
        edt),
    columns=['CCI'])
data10.to_csv("CCI_index.csv")
plt.figure(figsize=(20, 9))
plt.grid(True)
plt.plot(data10.index, data10["CCI"])
plt.title('Consumer confidence index - monthly')
plt.xlabel('Date')
plt.ylabel('Normalised (Normal=100)')
plt.savefig("CCI_index")
