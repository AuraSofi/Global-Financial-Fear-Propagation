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

"""

pairs = {
    "VIX_VHSI_26W": ("VIX Index", "VHSI Index"),
    "VIX_IVIUK_26W": ("VIX Index", "IVIUK Index"),
    "VIX_VXJ_26W": ("VIX Index", "VXJ Index"),
    "VXJ_VHSI_26W": ("VXJ Index", "VHSI Index"),
    "IVIUK_VXJ_26W": ("IVIUK Index", "VXJ Index"),
    "IVIUK_VHSI_26W": ("IVIUK Index", "VHSI Index")
}



rolling_correlations = pd.DataFrame()

for name, (index1, index2) in pairs.items():
    
    rolling_correlations[name] = func.roll_corr(df_global, index1, index2, 26)



#rolling_correlations.to_csv("/Users/aurasofi/Downloads/rolling_correlations.csv")



# ---------------------------------------------------------------------
# MAXIMUM CORRELATION DURING EACH CRISIS 
#   -Pico de correlacion entre ese rango de fechas
# ---------------------------------------------------------------------

print(rolling_correlations.idxmax())

for name, (start, end, color) in crisis.items():

    print(f"\n{name}")
    print(func.get_max_corr( rolling_correlations, start, end))




# ### por curiosidad- top 5 picos de corr en  cada serie
# for col in rolling_correlations.columns:
#     print(f"\n{col}")
#     print(rolling_correlations.nlargest(5, col)[col])




# -------------------------------------------------------------------
# DIFFERENCES IN ROLLING CORRELATIONS
# -------------------------------------------------------------------

diff_all = rolling_correlations.diff()

diff_crisis = func.get_crisis_differences(rolling_correlations, crisis)
    

# -------------------------------------------------------------------
# STATISTICS
# -------------------------------------------------------------------

statistics = func.get_diff_statistics (diff_all, diff_crisis)
#statistics.to_csv("/Users/aurasofi/Downloads/Statistics_roll_corr.csv")

print("\nStatistics of differences full period and during crises:")
print(statistics)    



"""
------------------------  CROSS - Correlation  ------------------------------ 

"""

cross_correlations = {}

for name, (index1, index2) in pairs.items():

    cross_correlations[name] = func.cross_corr(df_global[index1], df_global[index2], max_lag=12)


cross_correlations_df = pd.DataFrame(cross_correlations)


cross_correlations_df.to_csv("/Users/aurasofi/Downloads/cross_corr.csv")

for name, cc in cross_correlations.items():
    func.cc_gp(cc, name)
    
    
    
cross_corr_cri = {}

for crisis_name, (start, end, color) in crisis.items():
    
    df_crisis = df_global.loc[start:end]
    
    for pair_name, (index1, index2) in pairs.items():
        
        cross_corr_cri[f"{crisis_name}_{pair_name}"] = func.cross_corr(
            df_crisis[index1],
            df_crisis[index2],
            max_lag=12
        )

cross_correlations_df = pd.DataFrame(cross_corr_cri)

for name, cc in cross_corr_cri.items():
    func.cc_gp(cc, name)
    
    
    
    
    
    
    
    