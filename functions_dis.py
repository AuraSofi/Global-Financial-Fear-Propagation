#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 00:02:59 2026

@author: aurasofi
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

import seaborn as sns




def corr_pd(df, start_date, end_date):
    period = df.loc[start_date:end_date]
    corr = period.corr()
    
    print(f"Correlation matrix: {start_date} to {end_date}")
    print(corr)

    return corr

def corr_gp(corr, pd_name):
    
    plt.figure(figsize=(7,6))
    sns.heatmap(
        corr,
        annot=True,
        cmap='Blues',
        vmin=-1,
        vmax=1
    )
    
    plt.title(f"Correlation during Crisis: {pd_name}")
    plt.show()
    
    
def roll_corr(df, index1, index2, wk):
    rolling_corr = (
    df[index1]
    .rolling(wk)
    .corr(df[index2]))
    
    rolling_corr.plot(figsize=(14,5))
    plt.title(f"Rolling Correlation {index1} and {index2}" )
    plt.ylabel("Correlation")
    plt.show()
    
    

def cross_corr(x, y, max_lag):
    lags = range(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        corr = x.corr(y.shift(lag))
        correlations.append(corr)

    return pd.Series(correlations, index=lags)


def cc_gp(cross_corr):
    plt.figure(figsize=(10,5))
    plt.plot(cross_corr.index, cross_corr.values, marker="o")
    plt.axhline(0, color="green")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Lag (weeks)")
    plt.ylabel("Correlation")
    plt.title("Cross-correlation: VIX vs IVIUK")
    plt.grid(True)
    plt.show()