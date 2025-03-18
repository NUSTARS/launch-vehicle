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

project_root = Path(__file__).parent  # Gets the current directory where the script is located
data_dir = project_root / "data-2025" / "MPP" / "FRR" / "RAS_csvs"
fig_dir = project_root / "figures" / "MPP" / "FRR"  
###################################

# REQUIRES PYTHON 3.7

def format_ax(ax):
    pass

def standard_plot(dfs, file_names, fields, title):
    fig, ax = plt.subplots()
    for df in dfs:
        ax.plot(df[fields[0]], df[fields[1]])
    ax.set_title(title + ' for Possible Flight Paths')
    ax.set_xlabel(fields[0].replace("(", "[").replace(")", "]"))
    ax.set_ylabel(fields[1].replace("(", "[").replace(")", "]"))
    ax.legend(file_names)
    # DO NOT SHOW PLOT UNTIL THE MAIN FUNCTION

    return fig

def clipped_plot(dfs, file_names, fields, title, times):
    fig, ax = plt.subplots()
    for df in dfs:
        df = df.loc[(df[fields[0]] >= times[0]) & (df[fields[0]] <= times[1])]
        ax.plot(df[fields[0]], df[fields[1]])
    ax.set_title(title + ' for Possible Flight Paths')
    ax.set_xlabel(fields[0].replace("(", "[").replace(")", "]"))
    ax.set_ylabel(fields[1].replace("(", "[").replace(")", "]"))
    ax.legend(file_names)
    # DO NOT SHOW PLOT UNTIL THE MAIN FUNCTION

    return fig

def clipped_plot_2(fig, dfs, file_names, fields, title, times):
    ax = fig.axes[0]
    for df in dfs:
        df = df.loc[(df[fields[0]] >= times[0]) & (df[fields[0]] <= times[1])]
        ax.plot(df[fields[0]], df[fields[1]])
    ax.set_title(title + ' for Possible Flight Paths')
    ax.set_xlabel(fields[0].replace("(", "[").replace(")", "]"))
    ax.set_ylabel(fields[1].replace("(", "[").replace(")", "]"))
    ax.legend(file_names)

    return fig

def main():
    print("Hello, World!")
    dfs = []

    for file_name in file_names:     
        dfs.append(pd.read_csv(data_dir / f"{file_name}.csv", comment='#'))
        print(file_name + " Loaded")

    #fig = standard_plot(dfs, file_names, ['Time (sec)', 'Altitude (ft)'], "RasAero II Height AGL")
    #fig.axes[0].set_ylabel('Height AGL [ft]')
    #standard_plot(dfs, file_names, ['Time (sec)', 'Vel-V (ft/sec)'], "RasAero II Velocity")
    #standard_plot(dfs, file_names, ['Time (sec)', 'Accel-V (ft/sec^2)'], "RasAero II Acceleration")

    clipped_plot(dfs, file_names, ['Time (sec)', 'Stability Margin (cal)'], "RasAero II Dynamic Stability", [t_off_rail, 15])
    fig = clipped_plot(dfs, file_names, ['Time (sec)', 'CP (in)'], "RasAero II CG & CP Location from Nose Cone Tip", [t_off_rail, 15])
    fig = clipped_plot_2(fig, dfs, file_names, ['Time (sec)', 'CG (in)'], "RasAero II CG & CP Locations from Nose Cone Tip", [t_off_rail, 15])
    fig.axes[0].set_ylabel('Distance from Nose Cone Tip [in]')
    fig.axes[0].legend(['CP - Default', 'CP - Poor Conditons', 'CP - Ideal Conditons', 'CG - Default', 'CG - Poor Conditions', 'CG - Ideal Conditions'])

    if save_figures:
        for i in plt.get_fignums():
            print("saving...")
            plt.figure(i).savefig(fig_dir / f'{plt.figure(i).axes[0].get_title().replace(" ", "_")}.png', dpi=300)

    plt.show()

main()