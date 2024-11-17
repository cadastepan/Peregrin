# Peregrin

## Libraries
Importing **libraries** and abbreviating **functions**:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    import matplotlib.colors as mcolors
    from matplotlib.colors import Normalize, ListedColormap
    import matplotlib.colorbar as colorbar
    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Slider
    import matplotlib.animation as animation
    import matplotlib as mat
    import os
    import os.path as op
    from scipy.stats import gaussian_kde
    from matplotlib import font_manager as fm
    import seaborn as sea
    import statsmodels.api as sm
    from scipy.interpolate import make_interp_spline
    from scipy.stats import rayleigh, norm
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from scipy.signal import savgol_filter
    from peregrin.scripts import PlotParams, FilesAndFolders


## Input and Output paths, Files, Folders
Defining the data **input** - assigning the .csv data file's path to a variable "input_file". *If a folder on the Bryjalab server is selected, the user must be logged in while running the script. Also, just a reminder not to forget about activating the VPN connection when working from home ;)*

Defining the analysis's **output** - assigning a folder path to a variable "save_path" into which all the files created will be saved.

    # input file path:
    input_file = r"Z:\Shared\bryjalab\users\Branislav\Collagen Migration Assay DATA\data 23-7-24\run1\position_4!\C2-position_spots.csv"

    # save path:
    save_path = r"Z:\Shared\bryjalab\users\Branislav\Collagen Migration Assay DATA\data 23-7-24\run1\position_4!\analysed"

---

Loading the *.csv* **input file** as a DataFrame into the "df" variable:

    df = pd.read_csv(input_file)

---

*While working on the script, it may happen often that you'd like to modify the output directories, change titles or names of the files created and saved, or you might even fall out of love with certain plots, removing their code from the script completely. So with keeping this in mind, it is to be expected that constantly remodeling and rerunning the analysis on the same dataset will cause chaos.. For this reason, there is a **very sketchy function** you can use, if you dare :smiling_imp:.*

This ***very sketchy function* deletes all files** - though not directories *(folders):*

    FilesAndFolders.clean(folder)

---

