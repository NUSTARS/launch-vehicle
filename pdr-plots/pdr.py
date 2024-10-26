import pandas as pd
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
#from scipy.signal import savgol_filter
from PIL import Image
import matplotlib.ticker as ticker
import os
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")

###################################
file_names = ["Default", "Poor Conditions", "Ideal Conditions"]
save_figures = True    # Set this to true if you want to save the figures 
t_off_rail = 0.3  
###################################

# REQUIRES PYTHON 3.7

def format_ax(ax):
    pass

def standard_plot(dfs, names, x_field, y_field):
    fig, ax = plt.subplots()
    for df in dfs:
        ax.plot(df[x_field], df[y_field])
    ax.set_title(y_field + ' for Possible Flight Paths')
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.legend(names)
    # DO NOT SHOW PLOT UNTIL THE MAIN FUNCTION

def clipped_plot(dfs, names, x_field, y_field, start_time, end_time):
    fig, ax = plt.subplots()
    for df in dfs:
        df = df.loc[(df[x_field] >= start_time) & (df[x_field] <= end_time)]
        ax.plot(df[x_field], df[y_field])
    ax.set_title(y_field + ' for Possible Flight Paths')
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.legend(names)
    # DO NOT SHOW PLOT UNTIL THE MAIN FUNCTION

def main():
    print("Hello, World!")
    dfs = []

    for file_name in file_names:
        project_root = Path(__file__).parent  # Gets the current directory where the script is located
        data_dir = project_root / "OR_csvs"
        dfs.append(pd.read_csv(data_dir / f"{file_name}.csv", comment='#'))
        print(file_name + " Loaded")

    standard_plot(dfs, file_names, 'Time (s)', 'Altitude (ft)')
    standard_plot(dfs, file_names, 'Time (s)', 'Vertical velocity (ft/s)')
    standard_plot(dfs, file_names, 'Time (s)', 'Vertical acceleration (ft/s²)')
    clipped_plot(dfs, file_names, 'Time (s)', 'Stability margin calibers (​)', t_off_rail, 15)

    if save_figures:
        for i in plt.get_fignums():
            plt.figure(i).savefig(project_root / f'figures/figure_{i}.png', dpi=300)

    plt.show()

main()