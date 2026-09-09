#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 00:02:59 2026

@author: aurasofi
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from arch import arch_model
import numpy as np

from statsmodels.tsa.api import VAR

from hmmlearn import hmm

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
    
    
def crisis_correlation_statistics(rolling_correlations, crisis, pair_columns, pre_window=26):

    results = []

    for crisis_name, (start_date, end_date, color) in crisis.items():

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Crisis period
        crisis_data = rolling_correlations.loc[
            start_date:end_date,
            pair_columns
        ]

        # Position immediately before crisis
        start_pos = rolling_correlations.index.searchsorted(start_date)

        pre_start_pos = max(0, start_pos - pre_window)

        pre_crisis_data = rolling_correlations.iloc[
            pre_start_pos:start_pos
        ][pair_columns]

        for pair in pair_columns:

            crisis_series = crisis_data[pair].dropna()
            pre_series = pre_crisis_data[pair].dropna()

            if crisis_series.empty:
                continue

            mean_crisis = crisis_series.mean()
            median_crisis = crisis_series.median()
            max_crisis = crisis_series.max()
            std_crisis = crisis_series.std()

            mean_pre = pre_series.mean()

            change_from_pre = (
                mean_crisis - mean_pre
            )

            results.append({
                "Crisis": crisis_name,
                "Pair": pair,

                "Pre_Crisis_Mean": mean_pre,
                "Crisis_Mean": mean_crisis,
                "Crisis_Median": median_crisis,
                "Crisis_Max": max_crisis,
                "Crisis_SD": std_crisis,

                "Change_From_Pre": change_from_pre
            })

    return pd.DataFrame(results)
   
    
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
    
    
    
def summarize_cross_correlation(cross_correlations, pairs):
    
    results = []

    for name, cc in cross_correlations.items():

        # Obtener los dos índices del par
        index1, index2 = pairs[name]

        # Nombres limpios para presentación
        index1_clean = index1.replace(" Index", "")
        index2_clean = index2.replace(" Index", "")

        pair_label = f"{index1_clean}-{index2_clean}"

        # Lag donde la correlación es más fuerte en valor absoluto
        peak_lag = cc.abs().idxmax()

        # Mantener el signo original de la correlación
        peak_corr = cc.loc[peak_lag]

        # Interpretación del lead-lag
        # cross_corr usa: x.corr(y.shift(lag))
        if peak_lag > 0:
            leader = index2_clean
            follower = index1_clean

        elif peak_lag < 0:
            leader = index1_clean
            follower = index2_clean

        else:
            leader = "Contemporaneous"
            follower = "Contemporaneous"

        results.append({
            "Pair": pair_label,
            "Peak_Correlation": peak_corr,
            "Peak_Lag": peak_lag,
            "Leader": leader,
            "Follower": follower
        })

    return pd.DataFrame(results)
  
##############################################################################
    
    
    
# ==================================================
# NUEVA FUNCIÓN: ESTIMACIÓN GARCH + GJR-GARCH
# ==================================================


def estimate_garch_models(df_delta, indices=["VIX Index", "IVIUK Index", "VHSI Index", "VXJ Index"]):

    """
    Estima GARCH(1,1) y GJR-GARCH(1,1) para cada índice.

    Entrada:
        df_delta: DataFrame con primeras diferencias
                  de los índices de volatilidad.

    Salida:
        tabla_resultados
        volatilidad_condicional
        residuos_estandarizados
    """

    resultados = []

    volatilidad = pd.DataFrame(index=df_delta.index)
    residuos_std = pd.DataFrame(index=df_delta.index)

    for idx in indices:

        serie = df_delta[idx].dropna()
        
        # Modelo GARCH(1,1) - Persistencia
        modelo_garch = arch_model(serie, vol="Garch", p=1, q=1, dist="Normal")
        res_garch = modelo_garch.fit(disp="off")

        # Modelo GJR-GARCH(1,1) - Asimetría (o=1 = término asimétrico)
        modelo_gjr = arch_model(serie, vol="GARCH", p=1, o=1, q=1, dist="Normal")
        res_gjr = modelo_gjr.fit(disp="off")

        
        # Extraer parámetros
        omega  = res_garch.params["omega"]
        alpha  = res_garch.params["alpha[1]"]
        beta   = res_garch.params["beta[1]"]
        gamma  = res_gjr.params.get("gamma[1]", np.nan)
        persis = alpha + beta


        garch_aic = res_garch.aic
        garch_bic = res_garch.bic

        gjr_aic = res_gjr.aic
        gjr_bic = res_gjr.bic


        # Significancia del término asimétrico
        pval_gamma = res_gjr.pvalues.get("gamma[1]", 1)
        sig_gamma = "Significant (p < 0.05)" if pval_gamma < 0.05 else "Not significant (p ≥ 0.05)"

        # Guardar resultados
        resultados.append({
            "Index": idx,
            #"Omega(constante)": round(omega, 5),
            "Alpha (reacción)": round(alpha, 4),
            "Beta (inercia)": round(beta, 4),
            "GARCH_Persistence": round(persis, 4),
            "Gamma (asymmetric)": round(gamma, 4),
            "Gamma significant": sig_gamma,
            "Gamma_pvalue": round(pval_gamma, 4),
           
            "GARCH_AIC": garch_aic,
            "GARCH_BIC": garch_bic,
        
            "GJR_AIC": gjr_aic,
            "GJR_BIC": gjr_bic
        })

        # Guardar volatilidad condicional y residuos (para VAR y Spillovers)
        volatilidad[f"{idx}_sigma"] = res_garch.conditional_volatility
        residuos_std[f"{idx}_resid"] = res_garch.resid / res_garch.conditional_volatility


    # Devolver todo como DataFrames
    tabla = pd.DataFrame(resultados).set_index("Index")
    return tabla, volatilidad, residuos_std



# ==================================================
# FUNCIÓN: ESTIMACIÓN MODELO VAR
# ==================================================
def estimate_var_model(df_delta, max_lags=10):
    """
    Estima modelo VAR: selecciona retraso óptimo, estima, prueba causalidad.
    Entrada: df_delta → series en VARIACIONES (ya preparado por GARCH)
    Salida: (modelo_ajustado, resumen_causalidad, descomposición_varianza)
    """
    indices = df_delta.columns.tolist()
    
    # 1️ Seleccionar retraso óptimo
    modelo_var = VAR(df_delta)
    criterios = modelo_var.select_order(maxlags=max_lags)
    p_optimo = criterios.selected_orders['aic']  # AIC = criterio estándar
    print(f" Retraso óptimo seleccionado: p = {p_optimo} semanas")
    print(criterios.summary())
    
    lag_selection = pd.DataFrame({
    "Criterion": ["AIC", "BIC", "HQIC", "FPE"],
    "Selected_Lag": [
        criterios.selected_orders["aic"],
        criterios.selected_orders["bic"],
        criterios.selected_orders["hqic"],
        criterios.selected_orders["fpe"]
    ]})

    # 2️ Estimar VAR con p óptimo
    modelo_ajustado = modelo_var.fit(p_optimo)
    # print("\n" + "="*70)
    # print(" RESUMEN DEL MODELO VAR")
    # print("="*70)
    # print(modelo_ajustado.summary())
    
    print("\n" + "="*70)
    print("VAR STABILITY")
    print("="*70)
    
    print("Stable VAR:", modelo_ajustado.is_stable(verbose=True))

    # 3️ Prueba de Causalidad de Granger → ¿Quién lidera?
    print("\n" + "="*70)
    print(" CAUSALIDAD DE GRANGER (p-valor) — ¿Quién precede a quién?")
    print("="*70)
    causalidad = pd.DataFrame( np.nan, index=indices, columns=indices)
    
    for causa in indices:
        for efecto in indices:
            if causa == efecto:
                continue
            prueba = grangercausalitytests(df_delta[[efecto, causa]].dropna(), 
                                           maxlag=p_optimo, verbose=False)
            p_valor = prueba[p_optimo][0]['ssr_chi2test'][1]
            causalidad.loc[causa, efecto] = round(p_valor, 4)
            interpret = "Causa" if p_valor < 0.05 else "No causa"
            print(f"  {causa} → {efecto}: p = {round(p_valor,4)} | {interpret}")

    # 4️ Descomposición de varianza → INSUMO para Diebold-Yilmaz
    print("\n" + "="*70)
    print("DESCOMPOSICIÓN DE VARIANZA (horizonte = 12 semanas)")
    print("="*70)
    descomp = modelo_ajustado.fevd(12)  # 12 semanas = medio plazo
    print(descomp.summary())
    
    fevd_array = descomp.decomp
    
    fevd_12 = pd.DataFrame(
    fevd_array[:, -1, :],
    index=indices,
    columns=indices
    ) * 100
    
    fevd_12.index.name = "Response"
    fevd_12.columns.name = "Shock"

    return modelo_ajustado, lag_selection, causalidad, descomp, fevd_12
    


# ==================================================
# FUNCIÓN: GRÁFICOS DE IMPULSO-RESPUESTA
# ==================================================

    
def graficar_impulso_respuesta(modelo, nombre_emisor, horizonte=24):
    """
    Graficar funciones de impulso-respuesta ante un shock del emisor.
    """
    irf = modelo.irf(horizonte)
    orden = modelo.names
    idx_emisor = orden.index(nombre_emisor)

    # Graficar TODAS las respuestas ante el shock del emisor
    irf.plot(
        impulse=idx_emisor,
        figsize=(12, 7)
    )
    
    plt.suptitle(f"Respuesta ante un shock de {nombre_emisor}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    

# ==================================================
# FUNCIÓN: CALCULAR SPILLOVERS DIEBOLD-YILMAZ (2012)
# Descomposición GENERALIZADA (Pesaran & Shin 1998)
# ==================================================

def calculate_diebold_yilmaz_spillovers(modelo_var, horizonte=12):
    """
    Calcula el índice de spillovers Diebold-Yilmaz desde un modelo VAR estimado.
    Entrada: modelo VAR de statsmodels, horizonte H en semanas
    Salida: (tabla_spillovers, spillovers_direccionales, indice_total)
    """
    # Extraer componentes del VAR
    p = modelo_var.k_ar       # número de rezagos
    K = modelo_var.neqs   # número de variables
    nombres = modelo_var.names
    Sigma = modelo_var.sigma_u  # matriz de covarianza de residuos

    # 1 Calcular representación MA(∞) → matriz de coeficientes Ψ_h
    A = modelo_var.coefs       # (p, K, K) coeficientes A_1 ... A_p
    Psi = np.zeros((horizonte, K, K))
    Psi[0] = np.eye(K)         # Ψ_0 = I

    for h in range(1, horizonte):
        for m in range(min(p, h)):
            Psi[h] += A[m] @ Psi[h - 1 - m]

    # 2️ Descomposición de varianza de error GENERALIZADA (Pesaran & Shin, 1998)
    # θ_ij^g(h) = (σ_jj^-1 * (e_i' Ψ_h Σ e_j)^2) / (e_i' Ψ_h Σ Ψ_h' e_i)
    sigma_diag = np.diag(Sigma)
    inv_sigma_jj = 1.0 / sigma_diag

    theta_generalizada = np.zeros((K, K))

    for i in range(K):
        numerador = np.zeros(K)
        denominador = 0.0
        for h in range(horizonte):
            Psi_Sigma = Psi[h] @ Sigma
            ei = np.zeros(K)
            ei[i] = 1.0
            ei_Psi_Sigma = ei @ Psi_Sigma  # fila i de Ψ_h Σ

            # Acumular numerador y denominador
            for j in range(K):
                numerador[j] += inv_sigma_jj[j] * (ei_Psi_Sigma[j] ** 2)
            denominador += ei @ Psi[h] @ Sigma @ Psi[h].T @ ei.T

        theta_generalizada[i] = numerador / denominador

    # 3️ Normalizar filas → cada fila suma a 1
    tabla = theta_generalizada / theta_generalizada.sum(axis=1, keepdims=True)

    # 4️ Calcular spillovers direccionales
    propios = np.diag(tabla)
    de_otros = tabla.sum(axis=1) - propios       # Lo que RECIBE cada mercado
    a_otros = tabla.sum(axis=0) - propios         # Lo que EMITE cada mercado
    neto = a_otros - de_otros                     # Saldo: + emisor, - receptor

    # Índice total de spillovers
    indice_total = 100 * de_otros.sum() / K

    # 5️  tablas 
    tabla_spillovers = pd.DataFrame(np.round(tabla * 100, 2),
                                     index=nombres, columns=nombres)
    tabla_spillovers["FROM Others"] = np.round(de_otros * 100, 1)

    fila_emite = pd.DataFrame({"TO Others": np.round(a_otros * 100, 1),
                                "NET Spillover": np.round(neto * 100, 1)},
                               index=nombres)


    return tabla_spillovers, fila_emite, indice_total


# ==================================================
# FUNCIÓN: MODELO HMM — 3 REGÍMENES + SEÑALES TEMPRANAS
# ==================================================

def estimate_hmm_regimes(df_delta, volatilidad, rolling_corr, n_regimes=3):
    """
    Estima regímenes ocultos:
    Calma / Transición / Crisis

    Utiliza indicadores que varían a través del tiempo.
    """

    # ==================================================
    # 1. ALINEAR LAS SERIES TEMPORALES
    # ==================================================

    fecha_comun = (
        df_delta.index
        .intersection(volatilidad.index)
        .intersection(rolling_corr.index)
    )

    # ==================================================
    # 2. CONSTRUIR MATRIZ DE INDICADORES
    # ==================================================

    X = pd.DataFrame(index=fecha_comun)

    # Volatilidades condicionales GARCH
    X["VIX_vol"] = volatilidad.loc[
        fecha_comun, "VIX Index_sigma"
    ]

    X["IVIUK_vol"] = volatilidad.loc[
        fecha_comun, "IVIUK Index_sigma"
    ]

    X["VHSI_vol"] = volatilidad.loc[
        fecha_comun, "VHSI Index_sigma"
    ]

    X["VXJ_vol"] = volatilidad.loc[
        fecha_comun, "VXJ Index_sigma"
    ]

    
    X["corr_promedio"] = rolling_corr.loc[
        fecha_comun,
        "Average_Correlation"
    ]

    # Eliminar solamente observaciones realmente incompletas
    X = X.dropna()

    print("\nNúmero de observaciones para HMM:", len(X))
    print("Desde:", X.index.min())
    print("Hasta:", X.index.max())
    print("\nNaNs:")
    print(X.isna().sum())

    # ==================================================
    # 3. ESTANDARIZAR
    # ==================================================

    X_est = (X - X.mean()) / X.std()

    # Seguridad
    X_est = X_est.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    
    # ==================================================
    # 4. ESTIMAR HMM CON MÚLTIPLES INICIALIZACIONES
    # ==================================================
    
    best_model = None
    best_score = -np.inf
    best_seed = None
    
    for seed in range(20):
    
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=seed
        )
    
        model.fit(X_est)
    
        score = model.score(X_est)
    
        if score > best_score:
            best_score = score
            best_model = model
            best_seed = seed
    
    
    modelo_hmm = best_model
    
    print("\nBest HMM initialization:")
    print("Seed:", best_seed)
    print("Log-likelihood:", round(best_score, 2))

    # ==================================================
    # 5. PROBABILIDADES DE LOS ESTADOS OCULTOS
    # ==================================================

    probabilidades = modelo_hmm.predict_proba(X_est)

    df_prob = pd.DataFrame(
        probabilidades,
        index=X_est.index,
        columns=[
            f"Estado_{i}"
            for i in range(n_regimes)
        ]
    )

    # ==================================================
    # 6. IDENTIFICAR CALMA / TRANSICIÓN / CRISIS
    # ==================================================

    media_vix_regimen = []

    for r in range(n_regimes):

        media = np.average(
            X.loc[X_est.index, "VIX_vol"],
            weights=probabilidades[:, r]
        )

        media_vix_regimen.append(media)
        
        
    print("\nMean VIX volatility by hidden state:")
    print(media_vix_regimen)

    # menor volatilidad -> calma
    # mayor volatilidad -> crisis
    orden = np.argsort(media_vix_regimen)

    mapeo = {
        f"Estado_{orden[0]}": "🟢 Calma",
        f"Estado_{orden[1]}": "🟡 ALERTA / Transición",
        f"Estado_{orden[2]}": "🔴 CRISIS"
    }

    df_prob = df_prob.rename(columns=mapeo)

    # Ordenar las columnas
    df_prob = df_prob[
        [
            "🟢 Calma",
            "🟡 ALERTA / Transición",
            "🔴 CRISIS"
        ]
    ]
    
    # ==================================================
    # 7. RÉGIMEN DOMINANTE EN CADA FECHA
    # ==================================================
    
    df_prob["Regime"] = df_prob[
        [
            "🟢 Calma",
            "🟡 ALERTA / Transición",
            "🔴 CRISIS"
        ]
    ].idxmax(axis=1)
    
    # ==================================================
    # 8. MATRIZ DE TRANSICIÓN
    # ==================================================
    
    # Reordenar la matriz original del HMM para que siga:
    # Calma -> Transición -> Crisis
    
    transition_matrix_ordered = modelo_hmm.transmat_[np.ix_(orden, orden)]
    
    regime_names = [
        "🟢 Calma",
        "🟡 ALERTA / Transición",
        "🔴 CRISIS"
    ]
    
    transition_matrix = pd.DataFrame(
        transition_matrix_ordered,
        index=regime_names,
        columns=regime_names
    )
    
    return df_prob, modelo_hmm, X, transition_matrix



def detectar_transiciones_regimen(df_prob):
    """
    Detecta cambios en el régimen dominante del HMM.

    Entrada:
        df_prob: DataFrame que contiene la columna "Regime"

    Salida:
        DataFrame con:
        - fecha de transición
        - régimen anterior
        - régimen nuevo
        - tipo de transición
    """

    df = df_prob.copy()

    # Régimen de la semana anterior
    df["Previous_Regime"] = df["Regime"].shift(1)

    # Quedarnos únicamente con las fechas donde hubo cambio de régimen
    transitions = df[
        (df["Regime"] != df["Previous_Regime"]) &
        (df["Previous_Regime"].notna())
    ].copy()

    # Crear etiqueta de transición
    transitions["Transition"] = (
        transitions["Previous_Regime"]
        + " → "
        + transitions["Regime"]
    )

    # Dejar solamente columnas relevantes
    transitions = transitions[
        [
            "Previous_Regime",
            "Regime",
            "Transition"
        ]
    ]

    return transitions



## ==================================================
## FUNCIÓN: DETECTAR SEÑALES TEMPRANAS ANTES DE CADA CRISIS 
## ==================================================
def analyze_pre_crisis_indicators(
        early_warning_data,
        crisis_entries,
        window=12
    ):
    """
    Analiza el comportamiento de indicadores dinámicos
    antes de cada entrada al régimen de Crisis.

    Compara la primera mitad y la segunda mitad
    de una ventana previa de 12 observaciones.

    También calcula el lead time desde la entrada
    más reciente a Transition hasta Crisis.
    """

    results = []

    for crisis_date, row in crisis_entries.iterrows():

        # --------------------------------------------------
        # 1. LOCALIZAR LA FECHA DE CRISIS
        # --------------------------------------------------

        if crisis_date not in early_warning_data.index:
            continue

        pos = early_warning_data.index.get_loc(crisis_date)

        # Necesitamos suficientes observaciones previas
        if pos < window:
            continue


        # --------------------------------------------------
        # 2. VENTANA PREVIA
        # --------------------------------------------------

        # Importante:
        # NO incluimos la observación de Crisis
        pre_window = early_warning_data.iloc[
            pos-window:pos
        ].copy()


        # Dividir 12 observaciones:
        # primeras 6 vs últimas 6

        half = window // 2

        first_half = pre_window.iloc[:half]
        last_half = pre_window.iloc[half:]


        # --------------------------------------------------
        # 3. P(TRANSITION)
        # --------------------------------------------------

        p_transition_first = (
            first_half["P_Transition"].mean()
        )

        p_transition_last = (
            last_half["P_Transition"].mean()
        )

        delta_p_transition = (
            p_transition_last
            - p_transition_first
        )


        # --------------------------------------------------
        # 4. CORRELACIÓN
        # --------------------------------------------------

        corr_first = (
            first_half["Average_Correlation"].mean()
        )

        corr_last = (
            last_half["Average_Correlation"].mean()
        )

        delta_corr = (
            corr_last
            - corr_first
        )


        # --------------------------------------------------
        # 5. CONDITIONAL VOLATILITY
        # --------------------------------------------------

        vol_first = (
            first_half["Average_Volatility"].mean()
        )

        vol_last = (
            last_half["Average_Volatility"].mean()
        )

        delta_vol = (
            vol_last
            - vol_first
        )
   

        # --------------------------------------------------
        # 6. LEAD TIME DEL EPISODIO TRANSITION -> CRISIS
        # --------------------------------------------------
        
        previous_regime = row["Previous_Regime"]
        
        if previous_regime == "🟡 ALERTA / Transición":
        
            # Comenzamos una observación antes de Crisis
            transition_start_pos = pos - 1
        
            # Retroceder mientras sigamos en Transition
            while (
                transition_start_pos > 0
                and early_warning_data.iloc[
                    transition_start_pos - 1
                ]["Regime"] == "🟡 ALERTA / Transición"
            ):
                transition_start_pos -= 1
        
            transition_start_date = (
                early_warning_data.index[
                    transition_start_pos
                ]
            )
        
            lead_observations = (
                pos - transition_start_pos
            )
        
            lead_days = (
                crisis_date
                - transition_start_date
            ).days
        
        else:
        
            transition_start_date = pd.NaT
            lead_observations = 0
            lead_days = 0
            
    
        # --------------------------------------------------
        # 7. GUARDAR
        # --------------------------------------------------

        results.append({

            "Crisis_Date": crisis_date,

            "Previous_Regime": row["Previous_Regime"],

            "P_Transition_First":  p_transition_first,

            "P_Transition_Last":  p_transition_last,

            "Delta_P_Transition": delta_p_transition,

            "Correlation_First": corr_first,

            "Correlation_Last": corr_last,

            "Delta_Correlation": delta_corr,

            "Volatility_First": vol_first,

            "Volatility_Last": vol_last,

            "Delta_Volatility": delta_vol,

            "Transition_Start_Date": transition_start_date,

            "Lead_Observations": lead_observations,

            "Lead_Days": lead_days
        })


    return pd.DataFrame(results)

def calcular_duracion_regimenes(df_prob):

    df = df_prob.copy()

    # Identificar cada bloque consecutivo de un régimen
    df["Regime_Block"] = (
        df["Regime"] != df["Regime"].shift()
    ).cumsum()

    duration = (
        df.groupby("Regime_Block")
        .agg(
            Regime=("Regime", "first"),
            Start_Date=("Regime", lambda x: x.index.min()),
            End_Date=("Regime", lambda x: x.index.max()),
            Observations=("Regime", "size")
        )
        .reset_index(drop=True)
    )

    return duration


def create_pre_crisis_window(
        early_warning_data,
        crisis_entries,
        window=12
    ):

    events = []

    for crisis_date in crisis_entries.index:

        if crisis_date not in early_warning_data.index:
            continue

        pos = early_warning_data.index.get_loc(crisis_date)

        if pos < window:
            continue

        # Incluye las 12 observaciones anteriores
        # y la observación de entrada a Crisis
        event_window = early_warning_data.iloc[
            pos-window:pos+1
        ].copy()

        event_window["Event_Time"] = range(-window, 1)

        event_window["Crisis_Date"] = crisis_date

        events.append(event_window)

    event_data = pd.concat(events)

    return event_data