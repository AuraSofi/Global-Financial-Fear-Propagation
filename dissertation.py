#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:55:21 2026

@author: aurasofi
"""

import functions_dis as func

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss


df_VIX = pd.read_csv("/Users/aurasofi/Downloads/VIX.csv")
df_IVIUK = pd.read_csv("/Users/aurasofi/Downloads/IVIUK.csv")   #corregir
df_VHSI = pd.read_csv("/Users/aurasofi/Downloads/VHSI.csv")
df_VXJ = pd.read_csv("/Users/aurasofi/Downloads/VXJ.csv")


# Convertir las fechas
for df in [df_VIX, df_IVIUK, df_VHSI, df_VXJ]:
    df["Date"] = pd.to_datetime(df["Date"])
    

df_m1 = (                   
    df_VIX
    .merge(df_IVIUK, on="Date", how="inner")
)


df_m1 = df_m1.merge(df_VHSI, on= "Date", how="inner") 


df_global = df_m1.merge(df_VXJ, on= "Date", how="inner") 

df_global["Date"] = pd.to_datetime(df_global["Date"])
df_global = df_global.set_index("Date")
df_global = df_global.sort_index()

    
#df_global.to_csv("/Users/aurasofi/Documents/df_global_data.csv")


print(df_global.describe())

print(df_global.skew())
print(df_global.kurtosis())



#############################################


for col in df_global.columns:
    result = adfuller(df_global[col].dropna())

    print(f"\n{col}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    


for col in df_global.columns:
    statistic, p_value, lags, critical_values = kpss(df_global[col], regression="c", nlags="auto")
    
    print(f"\n{col}")
    print(f"KPSS Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Lags: {lags}")
    
    
print("\nCritical Values:")
for key, value in critical_values.items():
    print(f"{key}: {value}")
    
    

    
"""
------------------------  Global Crisis Graphs ------------------------------ 

"""


colors = plt.cm.Paired.colors
#colors = plt.cm.Pastel1.colors
#colors = plt.cm.Set3.colors

fig, ax = plt.subplots(figsize=(15,6))

for col, color in zip(df_global.columns, colors):
    ax.plot(df_global.index, df_global[col], label=col, color=color)

ax.axvspan('2007-06-01','2009-01-01',
           color="dodgerblue", alpha=0.2,
           label='2008 Financial Crisis')

ax.axvspan('2020-01-01','2020-08-01',
           color='cyan', alpha=0.2,
           label='COVID Crisis')

ax.axvspan('2014-06-01','2015-10-01',
           color='steelblue', alpha=0.2,
           label='China Crash')

ax.axvspan('2010-01-01','2011-08-01',
           color='skyblue', alpha=0.2,
           label='European Debt Crisis')

ax.axvspan('2022-01-01','2022-05-01',
           color='deepskyblue', alpha=0.2,
           label='Russia-Ukraine Crisis')

plt.title("Global Volatility Indices")
plt.xlabel("Date")
plt.ylabel("Volatility Index")
plt.grid(True)
plt.legend()
plt.show()
    


# Crisis: (start, end, color)
crisis = {
    "2008 Financial Crisis": ("2007-06-01", "2009-01-01", "dodgerblue"),
    "European Debt Crisis": ("2010-01-01", "2011-08-01", "skyblue"),
    "China Crash": ("2014-06-01", "2015-10-01", "steelblue"),
    "COVID Crisis": ("2020-01-01", "2020-08-01", "cyan"),
    "Russia-Ukraine Crisis": ("2022-01-01", "2022-05-01", "deepskyblue")
}

# Columnas de índices
indices = ["VIX Index", "IVIUK Index", "VHSI Index", "VXJ Index"]

for col in indices:

    plt.figure(figsize=(14,6))

    plt.plot(df_global.index, df_global[col], label=col, linewidth=1.5, color = "grey")

    # Agregar todas las crisis
    for nombre, (inicio, fin, color) in crisis.items():
        plt.axvspan(inicio, fin,
                    color=color,
                    alpha=0.2,
                    label=nombre)

    plt.title(f"{col}")
    plt.xlabel("Date")
    plt.ylabel("Volatility Index")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    
"""
------------------------  Correlation in Crisis  ------------------------------ 

"""


    
crisis_2008 = func.corr_pd(df_global, "2007-06-01", "2009-01-01")
crisis_euro = func.corr_pd(df_global, "2010-01-01", "2011-08-01")
crisis_china = func.corr_pd(df_global, "2014-06-01", "2015-10-01")
crisis_cvd = func.corr_pd(df_global, "2020-01-01", "2020-08-01")
crisis_ruuk = func.corr_pd(df_global, "2022-01-01", "2022-05-01")
#norm = func.corr_pd(df_global, "2013", "2017")


func.corr_gp(crisis_2008, "2008")
func.corr_gp(crisis_china, "China")
func.corr_gp(crisis_cvd, "Covid")
func.corr_gp(crisis_euro, "European Debt")
func.corr_gp(crisis_ruuk, "Russia-Ukraine")
#func.corr_gp(norm, "Normal")


"""
------------------------  ROLLING Correlation  ------------------------------ 

CHECAR SEMANAS  Y HACER PRUEBAS CON OTRAS COMBINACIONES
"""
    

func.roll_corr(df_global, "VIX Index", "VHSI Index", 52)
func.roll_corr(df_global, "VIX Index", "VHSI Index", 26)
func.roll_corr(df_global, "VIX Index", "IVIUK Index", 26)
func.roll_corr(df_global, "VIX Index", "VXJ Index", 26)
func.roll_corr(df_global, "VXJ Index", "VHSI Index", 26)


"""
------------------------  CROSS - Correlation  ------------------------------ 

HACER CROSS PARA TODAS LAS COMBINACIONES 

"""
vix = df_global["VIX Index"]
iviuk = df_global["IVIUK Index"]
vhsi = df_global["VHSI Index"]
vxj = df_global["VXJ Index"]

cc_vix_iviuk = func.cross_corr(vix,iviuk, max_lag=12)
cc_vix_vshi = func.cross_corr(vix, vhsi, max_lag=12)
cc_vix_vxj = func.cross_corr(vix, vxj, max_lag=12)

cc_iviuk_vshi = func.cross_corr(iviuk, vhsi, max_lag=12)
cc_iviuk_vxj = func.cross_corr(iviuk, vxj, max_lag=12)

cc_vshi_vxj = func.cross_corr(vhsi, vxj, max_lag=12)


func.cc_gp(cc_vix_iviuk)
func.cc_gp(cc_vix_vshi)
func.cc_gp(cc_vix_vxj)

func.cc_gp(cc_iviuk_vshi)
func.cc_gp(cc_iviuk_vxj)

func.cc_gp(cc_vshi_vxj)

#test 2
