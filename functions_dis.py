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


crisis = {
    "2008 Financial Crisis": ("2007-06-01", "2009-01-01", "dodgerblue"),
    "European Debt Crisis": ("2010-01-01", "2011-08-01", "skyblue"),
    "China Crash": ("2014-06-01", "2015-10-01", "steelblue"),
    "COVID Crisis": ("2020-01-01", "2020-08-01", "cyan"),
    "Russia-Ukraine Crisis": ("2022-01-01", "2022-05-01", "deepskyblue")
}



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
    rolling_corr = (df[index1].rolling(wk).corr(df[index2]))
    
    rolling_corr.plot(figsize=(14,5))
    
    # Agregar todas las crisis
    for nombre, (inicio, fin, color) in crisis.items():
        plt.axvspan(inicio, fin,
                    color=color,
                    alpha=0.2,
                    label=nombre)
    
    
    plt.title(f"Rolling Correlation {index1} and {index2}" )
    plt.ylabel("Correlation")
    plt.show()
    
    return rolling_corr


def get_max_corr(df, start_date, end_date):
    period = df.loc[start_date:end_date]

    return pd.DataFrame({
        "max_correlation": period.max(),
        "date": period.idxmax(), 
    })


def get_diff_corr(df, start_date, end_date):
    diff = df.diff()
    period = diff.loc[start_date:end_date]
    return period




def get_crisis_differences(df, crisis):
    """
    Calculate the differences in rolling correlations
    for each crisis period
    """

    diff_crisis = {}

    for nombre, (inicio, fin, color) in crisis.items():
        diff_crisis[nombre] = get_diff_corr(df, inicio, fin)

    return diff_crisis



def get_diff_statistics(diff_all, diff_crisis):
    """
    Calculate summary statistics for the differences
    in rolling correlations for each crisis.
    """

    results = []
    
    # Full period
    for column in diff_all.columns:
        
        series = diff_all[column].dropna()
    
        results.append({
            "Period": "Full Period",
            "Pair": column,
            "Mean": series.mean(),
            "Mean Absolute": series.abs().mean(),
            "Positive %": (series > 0).mean() * 100,
            "Negative %": (series < 0).mean() * 100,
            "Maximum": series.max(),
            "Minimum": series.min()
        })

    for crisis_name, df in diff_crisis.items():

        for column in df.columns:

            series = df[column].dropna()

            results.append({
                "Period": crisis_name,
                "Pair": column,
                "Mean": series.mean(),
                "Mean Absolute": series.abs().mean(),
                "Positive %": (series > 0).mean() * 100,
                "Negative %": (series < 0).mean() * 100,
                "Maximum": series.max(),
                "Minimum": series.min()
            })

    return pd.DataFrame(results)




def cross_corr(x, y, max_lag):
    lags = range(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        corr = x.corr(y.shift(lag))
        correlations.append(corr)

    return pd.Series(correlations, index=lags)

    
    
    
def cc_gp(cross_corr, title):
    plt.figure(figsize=(10, 5))

    plt.plot(cross_corr.index, cross_corr.values, marker="o")

    plt.axhline(0, color="green")
    plt.axvline(0, color="red", linestyle="--")

    plt.xlabel("Lag (weeks)")
    plt.ylabel("Correlation")
    plt.title(f"Cross-correlation: {title}")
    plt.grid(True)

    plt.show()