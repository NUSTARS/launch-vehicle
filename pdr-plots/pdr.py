import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from PIL import Image
import matplotlib.ticker as ticker
import os
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")

def main():
    print("Hello, World!")

    file_name = "hello"
    project_root = Path(__file__).parent  # Gets the current directory where the script is located
    data_dir = project_root / "pdr-data-2025"
    df_primary = pd.read_csv(data_dir / f"{file_name}.csv")
    df_backup = pd.read_csv(data_dir / f"{file_name}.csv")


    save_figures = False
    if save_figures:
        for i in plt.get_fignums():
            plt.figure(i).savefig(f'/figures/figure_{i}.png', dpi=300)

    plt.show()