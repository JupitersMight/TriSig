import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from p_values import *
import matplotlib.pylab as pylab
import os

curr_dir = os.getcwd()


def heatmaps(z_vals, y_vals, title, save_name, path):
    df = pd.DataFrame(data=z_vals).T
    df.columns = y_vals
    df = df.drop(columns=[df.columns[0]])
    df = df.rename(index=lambda ind: n_10000[ind]).T

    params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)

    fig, ax = plt.subplots(figsize=(7,9))
    plt.title(title, fontsize="xx-large")
    sns.heatmap(df, cmap='Reds_r', yticklabels=True, ax=ax, cbar_kws={'label': 'p-value'})
    ax.set_ylabel("(|Y|, |Z|)", fontsize="xx-large", labelpad=20)
    ax.set_xlabel("|X|", fontsize="xx-large", labelpad=20)
    plt.savefig(path+"\\"+save_name+'.png', bbox_inches='tight', dpi=500, transparent=True)
    plt.close('all')


######################################################### Theoretical p-value heatmap plots #########################################################

# Granularity of the data
n_10000 = [n * (10000/500)for n in range(1,501)]

# Number of labels per variable
cases_titles = ["|L| = 3", "|L| = 5", "|L| = 7"]

# theoretical p-value results for the case where variables are independent and time slices are independent
cases_ind_ind = [p_values_n_10000_ind_ind_3, p_values_n_10000_ind_ind_5, p_values_n_10000_ind_ind_7]
save_files_ind_ind = ["p_values_n_10000_ind_ind_3", "p_values_n_10000_ind_ind_5", "p_values_n_10000_ind_ind_7"]

# theoretical p-value results for the case where variables are independent and time slices have temporal dependency
cases_ind_time = [p_values_n_10000_ind_time_3, p_values_n_10000_ind_time_5, p_values_n_10000_ind_time_7]
save_files_ind_time = ["p_values_n_10000_ind_time_3", "p_values_n_10000_ind_time_5", "p_values_n_10000_ind_time_7"]

# theoretical p-value results for the case where variables are dependent and time slices are independent
cases_dep_ind = [p_values_n_10000_dep_ind_3, p_values_n_10000_dep_ind_5, p_values_n_10000_dep_ind_7]
save_files_dep_ind = ["p_values_n_10000_dep_ind_3", "p_values_n_10000_dep_ind_5", "p_values_n_10000_dep_ind_7"]

# theoretical p-value results for the case where variables are independent and time slices are also independent
cases_dep_time = [p_values_n_10000_dep_time_3, p_values_n_10000_dep_time_5, p_values_n_10000_dep_time_7]
save_files_dep_time = ["p_values_n_10000_dep_time_3", "p_values_n_10000_dep_time_5", "p_values_n_10000_dep_time_7"]


for i in range(len(cases_ind_ind)):
    heatmaps(cases_ind_ind[i], size_dataset_independent, cases_titles[i], save_files_ind_ind[i], curr_dir)
    heatmaps(cases_ind_time[i], size_dataset_independent, cases_titles[i], save_files_ind_time[i], curr_dir)
    heatmaps(cases_dep_ind[i], size_dataset_dependent, cases_titles[i], save_files_dep_ind[i], curr_dir)
    heatmaps(cases_dep_time[i], size_dataset_dependent, cases_titles[i], save_files_dep_time[i], curr_dir)






































