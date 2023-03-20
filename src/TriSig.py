import pandas as pd
import numpy as np
import math
from collections import Counter
from scipy.special import betainc
from scipy.stats import binom
from math import comb
from decimal import *


def y_ind_z_ind(dataframes, tricluster):
    Z = len(dataframes)
    Y = len(dataframes[0].columns)
    cols = dataframes[0].columns

    filtered_dataframes = [dataframes[i] for i in tricluster["times"]]
    p = Decimal(1.0)
    # Para constantes
    for z in range(len(filtered_dataframes)):
        for col in tricluster["columns"]:

            total_observed_samples = list(filtered_dataframes[z][cols[col]])
            pattern_observed_samples = list(filtered_dataframes[z][cols[col]].iloc[tricluster["rows"]])
            pattern_val = Counter(pattern_observed_samples).most_common(1)[0][0]
            filtered_samples = list(filter(lambda val: val == pattern_val, total_observed_samples))
            p = p * Decimal((len(filtered_samples)/len(total_observed_samples)))

    # Adjust p
    p = p * Decimal(comb(Z, len(filtered_dataframes)))
    p = float(p * Decimal(comb(Y, len(tricluster["columns"]))))
    if p > 1:
        p = 1

    return betainc(
        len(tricluster["rows"]),
        len(dataframes[0][dataframes[0].columns[0]]),
        p
    )


def y_ind_z_dep(dataframes, tricluster):
    Z = len(dataframes)
    Y = len(dataframes[0].columns)

    filtered_dataframes = [dataframes[i] for i in tricluster["times"]]
    p = Decimal(1.0)
    # Para constantes
    for col in tricluster["columns"]:
        total_observed_samples = []
        pattern_observed_samples = []
        for x in range(len(filtered_dataframes[filtered_dataframes[0].columns[0]])):
            val = ""
            for z in range(len(filtered_dataframes)):
                df = filtered_dataframes[z]
                val += str(df.loc[x, list(df.columns)[col]])
            if x in tricluster["rows"]:
                pattern_observed_samples.append(val)
            total_observed_samples.append(val)
        pattern_val = Counter(pattern_observed_samples).most_common(1)[0][0]
        filtered_samples = list(filter(lambda val: val == pattern_val, total_observed_samples))
        p = p * Decimal((len(filtered_samples)/len(total_observed_samples)))

    # Adjust p
    p = p * Decimal(comb(Z, len(filtered_dataframes)))
    p = float(p * Decimal(comb(Y, len(tricluster["columns"]))))
    if p > 1:
        p = 1

    return betainc(
        len(tricluster["rows"]),
        len(dataframes[0][dataframes[0].columns[0]]),
        p
    )


def y_dep_z_ind(dataframes, tricluster):
    Z = len(dataframes)
    Y = len(dataframes[0].columns)
    cols = dataframes[0].columns

    filtered_dataframes = [dataframes[int(i)] for i in tricluster["times"]]
    p = Decimal(1.0)
    # Para constantes
    for df in filtered_dataframes:
        pattern_val = ""
        for i_col in range(0, len(tricluster["columns"])):
            pattern_val += str(df.iloc[int(tricluster["rows"][0]), int(tricluster["columns"][i_col])])
        new_vals = []

        for x_val in range(0, len(df[cols[int(tricluster["columns"][0])]])):
            aux_val = ""
            for i_col in range(0, len(tricluster["columns"])):
                aux_val += str(df.iloc[x_val][int(tricluster["columns"][i_col])])
            new_vals.append(aux_val)

        counter = 0
        for val in new_vals:
            if val == pattern_val:
                counter+=1

        p *= Decimal((counter/len(new_vals)))

    # Adjust p
    p = p * Decimal(comb(Z, len(filtered_dataframes)))
    p = float(p * Decimal(comb(Y, len(tricluster["columns"]))))
    if p > 1:
        p = 1

    return betainc(
        len(tricluster["rows"]),
        len(dataframes[0][dataframes[0].columns[0]]),
        p
    )


def y_ind_z_markov(dataframes, tricluster, real_dataset=False):
    Z = len(dataframes)
    Y = len(dataframes[0].columns)
    cols = dataframes[0].columns

    filtered_dataframes = [dataframes[i] for i in tricluster["times"]]

    if real_dataset:
        for z_i in range(len(filtered_dataframes)):
            df = filtered_dataframes[z_i]
            for y_i in range(len(tricluster["columns"])):
                minimum_tri = df.iloc[tricluster["rows"], tricluster["columns"][y_i]].min()
                maximum_tri = df.iloc[tricluster["rows"], tricluster["columns"][y_i]].max()
                filtered_dataframes[z_i].iloc[:,tricluster["columns"][y_i]] = filtered_dataframes[z_i].iloc[:,tricluster["columns"][y_i]].apply(lambda x: 1 if minimum_tri<x<maximum_tri else 0)

    # Initialize probability of each bicluster
    p_for_each_dimension = [Decimal(1.0) for i in tricluster["times"]]
    p_for_each_transition = [Decimal(1.0) for i in range(len(tricluster["times"])-1)]

    total_samples = len(filtered_dataframes[0][filtered_dataframes[0].columns[0]])

    # Calculate the probability in each Z of pattern
    for z in range(len(filtered_dataframes)):
        # get pattern value for each column
        pattern_df = filtered_dataframes[z]
        pattern_df = pattern_df.filter([filtered_dataframes[z].columns[i] for i in tricluster["columns"]], axis=1)
        pattern_df = pattern_df.iloc[tricluster["rows"]]

        col_val = {}
        for col in pattern_df.columns:
            col_val[col] = Counter(list(pattern_df[col])).most_common(1)[0][0]

        for col in tricluster["columns"]:
            p_x = len(list(filter(lambda val: val == col_val[cols[col]], list(filtered_dataframes[z].iloc[:,col]))))

            p_for_each_dimension[z] *= Decimal((p_x / total_samples))

    # Calculate the probability in each transition in Z
    z = 0
    while z < (len(filtered_dataframes) - 1):
        z_0 = dataframes[z]
        z+=1
        z_1 = dataframes[z]

        for col in tricluster["columns"]:
            total_observed_samples = []
            pattern_observed_samples = []
            for x in range(len(filtered_dataframes[0][filtered_dataframes[0].columns[0]])):
                val = str(z_0.iloc[x, col]) + str(z_1.iloc[x, col])
                total_observed_samples.append(val)
                if x in tricluster["rows"]:
                    pattern_observed_samples.append(val)

            pattern_val = Counter(pattern_observed_samples).most_common(1)[0][0]
            filtered_samples = list(filter(lambda val: val == pattern_val, total_observed_samples))
            p_for_each_transition[z-1] = p_for_each_transition[z-1] * Decimal((len(filtered_samples)/len(total_observed_samples)))

    #Pattern probability
    p = p_for_each_dimension[0]
    for z in range(len(filtered_dataframes)-1):
        p *= (p_for_each_transition[z]/p_for_each_dimension[z])

    # Adjusted pattern probability
    # Y
    p = p * Decimal(comb(Y, len(tricluster["columns"])))
    # Z
    p = float(p * Decimal((Z-len(tricluster["times"])+1)))
    if p > 1:
        p = 1

    return betainc(
        len(tricluster["rows"]),
        len(dataframes[0][dataframes[0].columns[0]]),
        p
    )


def y_dep_z_markov(dataframes, tricluster):
    Z = len(dataframes)
    Y = len(dataframes[0].columns)
    cols = dataframes[0].columns

    filtered_dataframes = [dataframes[i] for i in tricluster["times"]]
    # Initialize probability of each bicluster
    p_for_each_dimension = [Decimal(1.0) for i in tricluster["times"]]
    p_for_each_transition = [Decimal(1.0) for i in range(len(tricluster["times"])-1)]

    # get pattern value for each column
    pattern_df = filtered_dataframes[0].\
        filter([filtered_dataframes[0].columns[i] for i in tricluster["columns"]], axis=1)
    pattern_df = pattern_df.iloc[tricluster["rows"]]

    col_val = {}
    for col in pattern_df.columns:
        col_val[col] = Counter(list(pattern_df[col])).most_common(1)[0][0]

    # Calculate the probability in each Z of pattern
    for z in range(len(filtered_dataframes)):
        df = dataframes[z]
        pattern_val = ""
        for i_col in range(0, len(tricluster["columns"])):
            pattern_val += str(df[cols[tricluster["columns"][i_col]]].iloc[tricluster["rows"][0]])
        new_vals = []

        for x_val in range(0, len(df[cols[tricluster["columns"][0]]])):
            aux_val = ""
            for i_col in range(0, len(tricluster["columns"])):
                aux_val += str(df.iloc[x_val][cols[tricluster["columns"][i_col]]])
            new_vals.append(aux_val)

        counter = 0
        for val in new_vals:
            if val == pattern_val:
                counter+=1

        p_for_each_dimension[z] *= Decimal((counter/len(new_vals)))


    # Calculate the probability in each transition in Z
    z = 0
    while z < (len(filtered_dataframes) - 1):
        z_0 = dataframes[z]
        z+=1
        z_1 = dataframes[z]

        pattern_val_0 = ""
        pattern_val_1 = ""
        for i_col in range(0, len(tricluster["columns"])):
            pattern_val_0 += str(z_0[cols[tricluster["columns"][i_col]]].iloc[tricluster["rows"][0]])
            pattern_val_1 += str(z_1[cols[tricluster["columns"][i_col]]].iloc[tricluster["rows"][0]])
        new_vals_zo = []
        new_vals_z1 = []
        for x_val in range(0, len(z_0[cols[tricluster["columns"][0]]])):
            aux_val_0 = ""
            aux_val_1 = ""
            for i_col in range(0, len(tricluster["columns"])):
                aux_val_0 += str(z_0[cols[tricluster["columns"][i_col]]].iloc[x_val])
                aux_val_1 += str(z_1[cols[tricluster["columns"][i_col]]].iloc[x_val])
            new_vals_zo.append(aux_val_0)
            new_vals_z1.append(aux_val_1)

        counter = 0
        for i in range(0, len(new_vals_zo)):
            if new_vals_zo[i] == pattern_val_0 and new_vals_z1[i] == pattern_val_1:
                counter+=1

        p_for_each_transition[z-1] = p_for_each_transition[z-1] * Decimal((counter/len(new_vals_zo)))

    #Pattern probability
    p = p_for_each_dimension[0]
    for z in range(len(filtered_dataframes)-1):
        if p_for_each_dimension[z] == Decimal(0.0):
            p = Decimal(math.inf)
            break
        p*= (p_for_each_transition[z]/p_for_each_dimension[z])

    # Y
    p = p * Decimal(comb(Y, len(tricluster["columns"])))
    # Z
    p = float(p * Decimal((Z-len(tricluster["times"])+1)))
    if p > 1:
        p = 1

    return betainc(
        len(tricluster["rows"]),
        len(dataframes[0][dataframes[0].columns[0]]),
        p
    )


def hochberg_critical_value(p_values, false_discovery_rate=0.05):
    ordered_pvalue = sorted(p_values)

    critical_values = []
    for i in range(1,len(ordered_pvalue)+1):
        critical_values.append((i/len(ordered_pvalue)) * false_discovery_rate)

    critical_val = critical_values[0]
    for i in reversed(range(len(ordered_pvalue))):
        if ordered_pvalue[i] < critical_values[i]:
            critical_val = critical_values[i]
            break

    return critical_val


def bonferroni_correction(alpha, m):
    return alpha / m


