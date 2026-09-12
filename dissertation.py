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
df_IVIUK = pd.read_csv("/Users/aurasofi/Downloads/IVIUK.csv")   
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



"""
------------------------  Stationary Series Test ------------------------------ 

""" 


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
------------------------  Crisis Structure ------------------------------ 

"""    
    
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


pairs = {
    "VIX_VHSI_26W": ("VIX Index", "VHSI Index"),
    "VIX_IVIUK_26W": ("VIX Index", "IVIUK Index"),
    "VIX_VXJ_26W": ("VIX Index", "VXJ Index"),
    "VXJ_VHSI_26W": ("VXJ Index", "VHSI Index"),
    "IVIUK_VXJ_26W": ("IVIUK Index", "VXJ Index"),
    "IVIUK_VHSI_26W": ("IVIUK Index", "VHSI Index")
}    
    

df_delta = df_global[indices].diff().dropna()
    
"""
------------------------  Global Crisis Graphs ------------------------------ 

"""


colors = plt.cm.Paired.colors
#colors = plt.cm.Pastel1.colors
#colors = plt.cm.Set3.colors

fig, ax = plt.subplots(figsize=(15,6))

for col, color in zip(df_global.columns, colors):
    ax.plot(df_global.index, df_global[col], label=col, color=color)


#Dibujar los periodos de crisis usando el diccionario
for crisis_name, (start, end, color) in crisis.items():
    ax.axvspan(
        start,
        end,
        color=color,
        alpha=0.2,
        label=crisis_name
    )

ax.set_title("Global Implied Volatility Indices")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility Index")
ax.grid(True)
ax.legend()
plt.show()




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
full_period = func.corr_pd(df_global, "2005-01-04", "2025-12-19")


func.corr_gp(crisis_2008, "2008")
func.corr_gp(crisis_china, "China Crash")
func.corr_gp(crisis_cvd, "Covid")
func.corr_gp(crisis_euro, "European Debt")
func.corr_gp(crisis_ruuk, "Russia-Ukraine")
func.corr_gp(full_period, "Full Period")


"""
------------------------  ROLLING Correlation  ------------------------------ 

we need pairs dictionary

"""


rolling_correlations = pd.DataFrame()


for name, (index1, index2) in pairs.items():
    
    rolling_correlations[name] = func.roll_corr(df_global, index1, index2, 26)
    
    
pair_columns = list(pairs.keys())

rolling_correlations["Average_Correlation"] = (rolling_correlations[pair_columns].mean(axis=1))



print(rolling_correlations[list(pairs.keys()) + ["Average_Correlation"]].head(30))
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

diff_all = rolling_correlations[pair_columns].diff()
    
diff_crisis = func.get_crisis_differences(rolling_correlations[pair_columns], crisis)

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
    cross_correlations[name] = func.cross_corr(df_delta[index1], df_delta[index2], max_lag=12)
    #cross_correlations[name] = func.cross_corr(df_global[index1], df_global[index2], max_lag=12)


cross_correlations_df = pd.DataFrame(cross_correlations)


cross_correlations_df.to_csv("/Users/aurasofi/Downloads/cross_corr.csv")

for name, cc in cross_correlations.items():
    func.cc_gp(cc, name)
    
    
    
cross_corr_cri = {}

for crisis_name, (start, end, color) in crisis.items():
    
    df_crisis = df_delta.loc[start:end]
    
    for pair_name, (index1, index2) in pairs.items():
        
        cross_corr_cri[f"{crisis_name}_{pair_name}"] = func.cross_corr(
            df_crisis[index1],
            df_crisis[index2],
            max_lag=12
        ) 
        
        
        
cross_corr_summary = func.summarize_cross_correlation(
    cross_correlations,
    pairs
)

print("\n" + "=" * 70)
print("CROSS-CORRELATION LEAD-LAG SUMMARY")
print("=" * 70)

print(
    cross_corr_summary
    .round(2)
    .to_string(index=False)
)





cross_correlations_df = pd.DataFrame(cross_corr_cri)

for name, cc in cross_corr_cri.items():
    func.cc_gp(cc, name)
    
 
"""
------------------------  statistics -  Correlation  ------------------------------ 

"""    
    
crisis_corr_stats = func.crisis_correlation_statistics(rolling_correlations=rolling_correlations,
    crisis=crisis, pair_columns=pair_columns, pre_window=26)


print("\n" + "=" * 70)
print("ROLLING CORRELATION STATISTICS BY CRISIS")
print("=" * 70)

print( crisis_corr_stats.round(4).to_string(index=False))



crisis_corr_summary = (
    crisis_corr_stats
    .groupby("Crisis")
    .agg(
        Pre_Crisis_Mean=("Pre_Crisis_Mean", "mean"),
        Crisis_Mean=("Crisis_Mean", "mean"),
        #Crisis_Median=("Crisis_Median", "mean"),
        Crisis_Max=("Crisis_Max", "max"),
        Mean_Change=("Change_From_Pre", "mean"),
        Pairs_Increased=("Change_From_Pre", lambda x: (x > 0).sum())
    )
    .reset_index()
)

crisis_corr_summary["Percentage_Increased"] = (crisis_corr_summary["Pairs_Increased"] / len(pair_columns) * 100)


print("\n" + "=" * 70)
print("CROSS-MARKET SYNCHRONIZATION BY CRISIS")
print("=" * 70)

print(
    crisis_corr_summary
    .round(2)
    .to_string(index=False)
)
#####################################################################


    # ======================
    #  NUEVO: ESTIMAR GARCH
    # ======================
 
tabla_garch, volatilidad, residuos_std = (func.estimate_garch_models(df_delta))

    # ======================
    #  MOSTRAR RESULTADOS
    # ======================
print("=" * 70)
print(" RESULTADOS GARCH(1,1) y GJR-GARCH(1,1)")
print("=" * 70)
print(tabla_garch.to_string())
print("✅ Residuos estandarizados: residuos_std")
print("✅ Series en diferencias: df_delta")


vol_columns = [
    "VIX Index_sigma",
    "IVIUK Index_sigma",
    "VHSI Index_sigma",
    "VXJ Index_sigma"
]

volatilidad["Average_Volatility"] = (
    volatilidad[vol_columns].mean(axis=1)
)

    # ======================
    #  GRÁFICO VOLATILIDAD
    # ======================
    
fig, ax = plt.subplots(figsize=(12, 6))

for idx in indices:
     ax.plot(volatilidad.index, volatilidad[f"{idx}_sigma"], label=idx, alpha=0.7)
ax.set_title("Conditional Volatility — GARCH(1,1)", fontsize=13)
ax.legend(), ax.grid(alpha=0.3), plt.tight_layout(), plt.show()
    


# ==========  AHORA EJECUTAMOS EL VAR ==========
modelo_var, lag_selection, causalidad, descomp_var, fevd_12 = func.estimate_var_model(df_delta, max_lags=10)



# ==========  GRÁFICOS DE RESPUESTA (lo más importante) ==========
# ¿Qué pasa si sube el miedo en EE.UU.?
func.graficar_impulso_respuesta(modelo_var, "VIX Index")

# ¿Qué pasa si sube el miedo en UK?
func.graficar_impulso_respuesta(modelo_var, "IVIUK Index")

# ==========  GUARDAR RESULTADOS ==========

print("\n" + "=" * 70)
print("VAR LAG SELECTION")
print("=" * 70)
print(lag_selection.to_string(index=False))

print("\n" + "=" * 70)
print("GRANGER CAUSALITY P-VALUES")
print("=" * 70)
print(causalidad.round(4).to_string())

print("\n" + "=" * 70)
print("FEVD AT HORIZON 12")
print("=" * 70)
print(fevd_12.round(2).to_string())

print("\n VAR completado. Resultados listos para Diebold-Yilmaz")  
    
    

# ==========  AHORA CALCULAR SPILLOVERS ===============================


tabla_spillovers, fila_emite, indice_total = func.calculate_diebold_yilmaz_spillovers(
    modelo_var, horizonte=12  # 12 semanas = igual que tu FEVD anterior
)

# # ==========  MOSTRAR RESULTADOS ==========
print("\n" + "=" * 70)
print("DIEBOLD-YILMAZ SPILLOVER TABLE")
print("=" * 70)
print(tabla_spillovers.round(2).to_string())

print("\n" + "=" * 70)
print("DIRECTIONAL AND NET SPILLOVERS")
print("=" * 70)
print(fila_emite.round(2).to_string())

print("\nTotal Spillover Index:")
print(round(indice_total, 2), "%")
    
    
    
    
# ==================================================
#  PASO 4: AHORA LLAMAMOS AL HMM / MARKOV-SWITCHING
# ==================================================


df_prob, modelo_hmm, X_indicadores, transition_matrix = func.estimate_hmm_regimes(
    df_delta=df_delta,
    volatilidad=volatilidad,
    rolling_corr=rolling_correlations,
    n_regimes=3
)
    
    
print("\n" + "=" * 70)
print("MATRIZ DE TRANSICIÓN DEL HMM")
print("=" * 70)
print(transition_matrix.round(3))

print("\n" + "=" * 70)
print("RÉGIMEN DOMINANTE - ÚLTIMAS 20 OBSERVACIONES")
print("=" * 70)
print(df_prob[["Regime"]].tail(20))



regime_transitions = func.detectar_transiciones_regimen(df_prob)

print("\n" + "=" * 70)
print("TRANSICIONES DE RÉGIMEN DETECTADAS")
print("=" * 70)

print(regime_transitions)

print("\nNúmero total de transiciones:")
print(len(regime_transitions))

print("\nTipos de transiciones:")
print(regime_transitions["Transition"].value_counts())



# ==================================================
#  MOSTRAR RESULTADOS FINALES
# ==================================================
print("\n" + "="*70)
print(" PROBABILIDADES DE RÉGIMEN (últimas 10 semanas)")
print("="*70)
print(df_prob.tail(10).round(3).to_string())



# ==================================================
#  EARLY WARNING DATA
# ==================================================


early_warning_data = pd.DataFrame(index=df_prob.index)

early_warning_data["P_Calm"] = df_prob["🟢 Calma"]
early_warning_data["P_Transition"] = df_prob["🟡 ALERTA / Transición"]
early_warning_data["P_Crisis"] = df_prob["🔴 CRISIS"]

early_warning_data["Average_Correlation"] = (
    rolling_correlations["Average_Correlation"]
)

early_warning_data["Average_Volatility"] = (
    volatilidad["Average_Volatility"]
)

early_warning_data["Regime"] = df_prob["Regime"]

early_warning_data = early_warning_data.dropna()


print("\n EARLY WARNING DATA")
print(early_warning_data.head())
print(early_warning_data.describe())


# ==================================================
#  CRISIS ENTRIES 
# ==================================================

crisis_entries = regime_transitions[
    regime_transitions["Regime"] == "🔴 CRISIS"
].copy()

print("\nENTRIES INTO CRISIS")
print(crisis_entries)

print("\nNumber of crisis entries:", len(crisis_entries))





# ==================================================
#  PRE-CRISIS EARLY WARNING ANALYSIS
# ==================================================
 
pre_crisis_analysis = func.analyze_pre_crisis_indicators(
    early_warning_data=early_warning_data,
    crisis_entries=crisis_entries,
    window=12
)



# ==================================================
# RESULTADOS FINALES - EARLY WARNING SIGNALS
# ==================================================

print("\n" + "=" * 70)
print("EARLY WARNING SIGNALS BEFORE HIGH-STRESS REGIME")
print("=" * 70)

print(
    pre_crisis_analysis
    .round(4)
    .to_string(index=False)
)


# ==================================================
# RESUMEN DE INDICADORES
# ==================================================

n_crises = len(pre_crisis_analysis)

summary_ews = pd.DataFrame({

    "Indicator": [
        "Transition Probability",
        "Average Correlation",
        "Conditional Volatility"
    ],

    "Episodes_Increased": [
        (pre_crisis_analysis["Delta_P_Transition"] > 0).sum(),
        (pre_crisis_analysis["Delta_Correlation"] > 0).sum(),
        (pre_crisis_analysis["Delta_Volatility"] > 0).sum()
    ]
})

summary_ews["Total_Episodes"] = n_crises

summary_ews["Percentage_Increased"] = (
    summary_ews["Episodes_Increased"]
    / summary_ews["Total_Episodes"]
    * 100
)

print("\n" + "=" * 70)
print("EARLY WARNING SUMMARY")
print("=" * 70)

print(
    summary_ews
    .round(2)
    .to_string(index=False)
)


# ==================================================
# LEAD TIME: TRANSITION -> CRISIS
# ==================================================

transition_crises = pre_crisis_analysis[
    pre_crisis_analysis["Previous_Regime"]
    == "🟡 ALERTA / Transición"
].copy()

print("\n" + "=" * 70)
print("TRANSITION REGIME LEAD TIME")
print("=" * 70)

print(
    transition_crises[
        [
            "Crisis_Date",
            "Transition_Start_Date",
            "Lead_Observations",
            "Lead_Days"
        ]
    ].to_string(index=False)
)


print("\nAverage lead observations:",
      round(transition_crises["Lead_Observations"].mean(), 2))

print("Median lead observations:",
      round(transition_crises["Lead_Observations"].median(), 2))

print("Average lead days:",
      round(transition_crises["Lead_Days"].mean(), 2))

print("Median lead days:",
      round(transition_crises["Lead_Days"].median(), 2))

print("\n" + "=" * 70)


# ==================================================
# TABLA FOR EARLY WARNING
# ==================================================

event_data = func.create_pre_crisis_window(
    early_warning_data,
    crisis_entries,
    window=12
)

event_summary = (
    event_data
    .groupby("Event_Time")
    [
        [
            "Average_Correlation",
            "Average_Volatility",
            "P_Transition"
        ]
    ]
    .mean()
)
print(event_summary.round(4))



# ==================================================
# GRAFICO synchronization
# ==================================================

plt.figure(figsize=(9, 5))

plt.plot(
    event_summary.index,
    event_summary["Average_Correlation"],
    marker="o"
)

plt.axvline(
    x=0,
    linestyle="--",
    label="Entry into Crisis regime"
)

plt.xlabel("Observations relative to Crisis regime entry")
plt.ylabel("Average rolling correlation")

plt.title(
    "Cross-Market Synchronization Before High-Stress Regime Entry"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




# ==================================================
# GRAFICO CONDITIONAL VOLATILITY
# ==================================================


plt.figure(figsize=(9, 5))

plt.plot(
    event_summary.index,
    event_summary["Average_Volatility"],
    marker="o"
)

plt.axvline(
    x=0,
    linestyle="--",
    label="Entry into Crisis regime"
)

plt.xlabel("Observations relative to Crisis regime entry")
plt.ylabel("Average conditional volatility")

plt.title(
    "Conditional Volatility Before High-Stress Regime Entry"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# GRAFICO Transition probability
# ==================================================

plt.figure(figsize=(9, 5))

plt.plot(
    event_summary.index,
    event_summary["P_Transition"],
    marker="o"
)

plt.axvline(
    x=0,
    linestyle="--",
    label="Entry into Crisis regime"
)

plt.xlabel("Observations relative to Crisis regime entry")
plt.ylabel("Probability of Transition regime")

plt.title(
    "Transition-Regime Probability Before High-Stress Regime Entry"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# GRAFICAR PROBABILIDADES DE RÉGIMEN EN EL TIEMPO
# ==================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Calma
axes[0].plot(
    df_prob.index,
    df_prob["🟢 Calma"],
    color="green",
    label="Calm",
    lw=2
)
axes[0].set_title("Probability: Calm Regime", fontsize=12)
axes[0].grid(alpha=0.3)

# Transición
axes[1].plot(
    df_prob.index,
    df_prob["🟡 ALERTA / Transición"],
    color="orange",
    label="Transition",
    lw=2
)
axes[1].set_title("Probability: Transition Regime", fontsize=12)
axes[1].grid(alpha=0.3)

# Crisis
axes[2].plot(
    df_prob.index,
    df_prob["🔴 CRISIS"],
    color="red",
    label="Crisis",
    lw=2
)
axes[2].set_title("Probability: Crisis Regime", fontsize=12)
axes[2].grid(alpha=0.3)

for i, ax in enumerate(axes):
    for crisis_name, (start, end, color) in crisis.items():
        ax.axvspan(
            start,
            end,
            color=color,
            alpha=0.15,
            label=crisis_name if i == 2 else None
        )


# ==================================================
# CRISIS WINDOWS
# ==================================================

for ax in axes:
    for crisis_name, (start, end, color) in crisis.items():
        ax.axvspan(
            start,
            end,
            color=color,
            alpha=0.15
        )


# Labels
axes[0].set_ylabel("Probability")
axes[1].set_ylabel("Probability")
axes[2].set_ylabel("Probability")
axes[2].set_xlabel("Date")

axes[0].legend()
axes[1].legend()
axes[2].legend()

plt.suptitle(
    "Evolution of HMM Regime Probabilities",
    fontsize=15,
    y=1.01
)

plt.tight_layout()
plt.show()






# ===========================================================================

# ==================================================
# DIAGNÓSTICO DEL HMM
# ==================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO DEL HMM")
print("TRANSITION MATRIX")
print("=" * 70)
print(transition_matrix.round(4))


# Número de observaciones clasificadas en cada régimen
print("\nNumber of observations per dominant regime:")
print(df_prob["Regime"].value_counts())


# Probabilidad promedio de cada régimen
prob_cols = [
    "🟢 Calma",
    "🟡 ALERTA / Transición",
    "🔴 CRISIS"
]

print("\nAverage regime probabilities:")
print(df_prob[prob_cols].mean())


# Probabilidad del régimen dominante
df_prob["Max_Probability"] = df_prob[prob_cols].max(axis=1)

print("\nDominant regime probability:")
print(df_prob["Max_Probability"].describe())


# Ver las primeras 30 observaciones
print("\nFirst 30 regime probabilities:")
print(
    df_prob[
        prob_cols + ["Regime", "Max_Probability"]
    ].head(30)
)


# ==================================================
# REGIME DURATION
# ==================================================
regime_duration = func.calcular_duracion_regimenes(df_prob)

print("\n" + "=" * 70)
print("Regime Duration")
print("REGIME DURATIONS")
print("=" * 70)

print(regime_duration.to_string(index=False))


duration_summary = (
    regime_duration
    .groupby("Regime")["Observations"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\nAverage regime duration:")
print(duration_summary.round(2))


# ==================================================
# CHARACTERISTICS OF EACH REGIME
# ==================================================

regime_characteristics = X_indicadores.copy()

regime_characteristics["Regime"] = df_prob.loc[
    regime_characteristics.index, "Regime"
]

regime_summary = (
    regime_characteristics
    .groupby("Regime")
    .mean()
)

print("\n" + "=" * 70)
print("AVERAGE CHARACTERISTICS BY REGIME")
print("=" * 70)

print(regime_summary.round(4))
    
    
    