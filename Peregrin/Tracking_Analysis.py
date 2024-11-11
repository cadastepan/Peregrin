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


# TOTAL_DISTANCE => TRACK_LENGTH


# INPUT FILE:
input_file = r"Z:\Shared\bryjalab\users\Branislav\Collagen Migration Assay DATA\data 23-7-24\run1\position_4!\C2-position_spots.csv"

# SAVE PATH:
save_path = r"Z:\Shared\bryjalab\users\Branislav\Collagen Migration Assay DATA\data 23-7-24\run1\position_4!\analysed"

# Function fro file deletion in a selected folder 
"""for filename in os.listdir(save_path): # deletion of all files in selected folder
    file_path = os.path.join(save_path, filename)
    try:
        # Check if it is a file and then delete it
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        # If it is a directory, skip it
        elif os.path.isdir(file_path):
            print(f"Skipped directory: {file_path}")
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {e}")
print("All files have been deleted.")"""


title_size = 18
label_size = 11
figtext_size = 9
compass_annotations_size = 15
figtext_color = 'grey'

# Load the data
df = pd.read_csv(input_file)

# Drop rows with non-numeric values and convert columns to numeric
df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['POSITION_X', 'POSITION_Y', 'POSITION_Z', 'POSITION_T'])

# Reflect y-coordinates around the midpoint
y_mid = (df['POSITION_Y'].min() + df['POSITION_Y'].max()) / 2
df['POSITION_Y'] = 2 * y_mid - df['POSITION_Y']

# Ensure the DataFrame is sorted by TRACK_ID and POSITION_T (time)
df = df.sort_values(by=['TRACK_ID', 'POSITION_T'])

# Load the data into a DataFrame
df = pd.DataFrame(df)

# Definition of micron length per pixel
microns_per_pixel = 0.7381885238402274 # for 10x lens

# Define the desired dimensions in microns
x_min, x_max = 0, (1600 * microns_per_pixel)
y_min, y_max = 0, (1200 * microns_per_pixel)

# Calculate the aspect ratio
aspect_ratio = x_max / y_max


unneccessary_float_columns = [ # Definition of unneccesary float columns in the df which are to be convertet to integers
    'ID', 
    'TRACK_ID', 
    'POSITION_T', 
    'FRAME'
    ]
def butter(df, float_columns): # Smoothing the raw dataframe

    # Reset the df index
    df = df.reset_index(drop=True)

    # Define columns for consistency
    df.columns = [
        'LABEL', 
        'ID', 
        'TRACK_ID', 
        'QUALITY', 
        'POSITION_X', 
        'POSITION_Y', 
        'POSITION_Z', 
        'POSITION_T', 
        'FRAME', 
        'RADIUS', 
        'VISIBILITY', 
        'MANUAL_SPOT_COLOR', 
        'MEAN_INTENSITY_CH1', 
        'MEDIAN_INTENSITY_CH1', 
        'MIN_INTENSITY_CH1', 
        'MAX_INTENSITY_CH1', 
        'TOTAL_INTENSITY_CH1', 
        'STD_INTENSITY_CH1', 
        'EXTRACK_P_STUCK', 
        'EXTRACK_P_DIFFUSIVE', 
        'CONTRAST_CH1', 
        'SNR_CH1'
        ]

    # Drop all NaN values, also drop columns with NaN values
    df = df.dropna(axis=1)

    # Conversion of unnecessary floats to integers
    df[float_columns] = df[float_columns].astype(int)

    return df

def calculate_traveled_distances_for_each_cell_per_frame(df):

    # Calculate distances between consecutive frames
    if df.empty:
            return np.nan
    
    def distance_per_frame(group):
        group = group.copy()
        # Compute the Euclidean distance between consecutive frames
        group['DISTANCE'] = np.sqrt(
            (group['POSITION_X'].diff() ** 2) +
            (group['POSITION_Y'].diff() ** 2)
        )
        return group
    
    # Ensure TRACK_ID is not used as index in grouping; Calculate distances; Drop rows where DISTANCE is NaN (first frame of each TRACK_ID group)
    result_df = df.reset_index(drop=True).groupby('TRACK_ID', as_index=False).apply(distance_per_frame).dropna(subset='DISTANCE')

    return pd.DataFrame(result_df)

def calculate_total_distance_traveled_for_each_cell(df):

    # Convert 'Track ID' to numeric (if it's not already)
    df['TRACK_ID'] = pd.to_numeric(df['TRACK_ID'], errors='coerce')
    
    # Making sure that no empty lines are created in the DataFrame
    if df.empty:
        return np.nan
    
    # Sum distances per Track ID
    total_distance_per_cell = df.groupby('TRACK_ID')['DISTANCE'].sum().reset_index()

    # Rename columns for clarity
    total_distance_per_cell.columns = ['TRACK_ID', 'TOTAL_DISTANCE']

    # Return the results
    return pd.DataFrame(total_distance_per_cell)

def calculate_net_distance_traveled_for_each_cell(df):

    # Convert 'TRACK_ID' to numeric (if it's not already)
    df['TRACK_ID'] = pd.to_numeric(df['TRACK_ID'], errors='coerce')

    # Get the start and end positions for each track, calculate the enclosed distance and group by 'TRACK_ID'
    def calculate_distance(start, end):
        return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    def net_distance_per_track(track_df):
        if track_df.empty:
            return np.nan
        start_position = track_df.iloc[0][['POSITION_X', 'POSITION_Y']].values
        end_position = track_df.iloc[-1][['POSITION_X', 'POSITION_Y']].values
        return calculate_distance(start_position, end_position)

    net_distances = df.groupby('TRACK_ID').apply(net_distance_per_track).reset_index(name='NET_DISTANCE')

    # Return the results
    return pd.DataFrame(net_distances)

def calculate_confinement_ratio_for_each_cell_and_aggregate_with_total_distances_net_distances(df):
    # Calculate the confinement ratio
    df['CONFINEMENT_RATIO'] = df['NET_DISTANCE'] / df['TOTAL_DISTANCE']
    Track_stats1_df = df[['TRACK_ID','CONFINEMENT_RATIO']]
    return pd.DataFrame(Track_stats1_df)

def calculate_distances_per_frame(df):
    min_distance_per_frame = df.groupby('POSITION_T')['DISTANCE'].min().reset_index()
    min_distance_per_frame.rename(columns={'DISTANCE': 'min_DISTANCE'}, inplace=True)
    max_distance_per_frame = df.groupby('POSITION_T')['DISTANCE'].max().reset_index()
    max_distance_per_frame.rename(columns={'DISTANCE': 'max_DISTANCE'}, inplace=True)
    mean_distances_per_frame = df.groupby('POSITION_T')['DISTANCE'].mean().reset_index()
    mean_distances_per_frame.rename(columns={'DISTANCE': 'MEAN_DISTANCE'}, inplace=True)
    std_deviation_distances_per_frame = df.groupby('POSITION_T')['DISTANCE'].std().reset_index()
    std_deviation_distances_per_frame.rename(columns={'DISTANCE': 'STD_DEVIATION_distances'}, inplace=True)
    median_distances_per_frame = df.groupby('POSITION_T')['DISTANCE'].median().reset_index()
    median_distances_per_frame.rename(columns={'DISTANCE': 'MEDIAN_DISTANCE'}, inplace=True)
    merge = pd.merge(min_distance_per_frame, max_distance_per_frame, on='POSITION_T')
    merge = pd.merge(merge, mean_distances_per_frame, on='POSITION_T')
    merge = pd.merge(merge, std_deviation_distances_per_frame, on='POSITION_T')
    merged = pd.merge(merge, median_distances_per_frame, on='POSITION_T')
    return pd.DataFrame(merged)

def calculate_direction_of_travel_for_each_cell_per_frame(df):

    directions = []
    for track_id in df['TRACK_ID'].unique():
        track_data = df[df['TRACK_ID'] == track_id]
        dx = track_data['POSITION_X'].diff().iloc[1:]
        dy = track_data['POSITION_Y'].diff().iloc[1:]
        rad = (np.arctan2(dy, dx))
        for i in range(len(rad)):
            directions.append({
                'TRACK_ID': track_id, 
                'POSITION_T': track_data['POSITION_T'].iloc[i + 1], 
                'DIRECTION_RAD': rad.iloc[i]
                })
    return pd.DataFrame(directions)

def calculate_directions_per_cell(df):
    mean_direction_rad = df.groupby('TRACK_ID')['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    std_deviation_rad = df.groupby('TRACK_ID')['DIRECTION_RAD'].apply(lambda angles: np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    std_deviatin_deg = np.degrees(std_deviation_rad) % 360
    median_direction_rad = df.groupby('TRACK_ID')['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    median_direction_deg = np.degrees(median_direction_rad) % 360
    return pd.DataFrame({
        'TRACK_ID': mean_direction_rad.index, 
        'MEAN_DIRECTION_DEG': mean_direction_deg, 
        'STD_DEVIATION_DEG': std_deviatin_deg, 
        'MEDIAN_DIRECTION_DEG': median_direction_deg, 
        'MEAN_DIRECTION_RAD': mean_direction_rad, 
        'STD_DEVIATION_RAD': std_deviation_rad, 
        'MEADIAN_DIRECTION_RAD': median_direction_rad
        }).reset_index(drop=True)

def calculate_absolute_directions_per_frame(df):
    grouped = df.groupby('POSITION_T')
    mean_direction_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    std_deviation_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    std_deviatin_deg = np.degrees(std_deviation_rad) % 360
    median_direction_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    median_direction_deg = np.degrees(median_direction_rad) % 360
    return pd.DataFrame({
        'POSITION_T': mean_direction_rad.index, 
        'MEAN_DIRECTION_DEG_abs': mean_direction_deg, 
        'STD_DEVIATION_DEG_abs': std_deviatin_deg, 
        'MEDIAN_DIRECTION_DEG_abs': median_direction_deg, 
        'MEAN_DIRECTION_RAD_abs': mean_direction_rad, 
        'STD_DEVIATION_RAD_abs': std_deviation_rad, 
        'MEADIAN_DIRECTION_RAD_abs': median_direction_rad
        }).reset_index(drop=True)

def calculate_weighted_directions_per_frame(df):
    def weighted_mean_direction(angles, weights):
        weighted_sin = np.average(np.sin(angles), weights=weights)
        weighted_cos = np.average(np.cos(angles), weights=weights)
        return np.arctan2(weighted_sin, weighted_cos)
    
    def weighted_std_deviation(angles, weights, mean_direction_rad):
        weighted_sin = np.average(np.sin(angles - mean_direction_rad), weights=weights)
        weighted_cos = np.average(np.cos(angles - mean_direction_rad), weights=weights)
        return np.sqrt(weighted_sin**2 + weighted_cos**2)
    
    def weighted_median_direction(angles, weights):
        # Weighted median approximation: using percentile function from scipy
        from scipy.stats import cumfreq
        sorted_angles = np.sort(angles)
        sorted_weights = np.array(weights)[np.argsort(angles)]
        cumsum_weights = np.cumsum(sorted_weights)
        midpoint = np.sum(weights) / 2
        idx = np.searchsorted(cumsum_weights, midpoint)
        return sorted_angles[idx]

    grouped = df.groupby('POSITION_T')
    
    # Compute weighted metrics
    mean_direction_rad = grouped.apply(lambda x: weighted_mean_direction(x['DIRECTION_RAD'], x['DISTANCE'])).reset_index(level=0, drop=True)
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    
    std_deviation_rad = grouped.apply(lambda x: weighted_std_deviation(x['DIRECTION_RAD'], x['DISTANCE'], weighted_mean_direction(x['DIRECTION_RAD'], x['DISTANCE']))).reset_index(level=0, drop=True)
    std_deviation_deg = np.degrees(std_deviation_rad) % 360
    
    median_direction_rad = grouped.apply(lambda x: weighted_median_direction(x['DIRECTION_RAD'], x['DISTANCE'])).reset_index(level=0, drop=True)
    median_direction_deg = np.degrees(median_direction_rad) % 360
    
    result_df = pd.DataFrame({
        'POSITION_T': mean_direction_rad.index,
        'MEAN_DIRECTION_DEG_weight': mean_direction_deg,
        'STD_DEVIATION_DEG_weight': std_deviation_deg,
        'MEDIAN_DIRECTION_DEG_weight': median_direction_deg,
        'MEAN_DIRECTION_RAD_weight': mean_direction_rad,
        'STD_DEVIATION_RAD_weight': std_deviation_rad,
        'MEDIAN_DIRECTION_RAD_weight': median_direction_rad
    }).reset_index(drop=True)

    result_df['POSITION_T'] = result_df['POSITION_T'] + 1
    
    return result_df

def calculate_turn_angles_for_each_cell_per_frame(df):
    # Sort the DataFrame by TRACK_ID and POSITION_T to maintain order
    df = df.sort_values(by=['TRACK_ID', 'POSITION_T'])
    
    # Function to compute the angle difference
    def compute_angle_difference(angles):
        # Compute the difference between consecutive angles
        diff = np.degrees(abs(np.diff(abs(angles))))
        return np.concatenate(([0], diff))  # Return the result with a 0 for the first position

    # Apply the function to each TRACK_ID group
    df['ANGLE_TURNED'] = df.groupby('TRACK_ID')['DIRECTION_RAD'].transform(compute_angle_difference)
    
    df = df[['TRACK_ID', 'POSITION_T', 'ANGLE_TURNED']]

    return df

def calculate_angles_per_frame(df):
    min_angle_per_frame = df.groupby('POSITION_T')['ANGLE_TURNED'].min().reset_index()
    min_angle_per_frame.rename(columns={'ANGLE_TURNED': 'min_ANGLE_TURNED'}, inplace=True)
    max_angle_per_frame = df.groupby('POSITION_T')['ANGLE_TURNED'].max().reset_index()
    max_angle_per_frame.rename(columns={'ANGLE_TURNED': 'max_ANGLE_TURNED'}, inplace=True)
    mean_angles_per_frame = df.groupby('POSITION_T')['ANGLE_TURNED'].mean().reset_index()
    mean_angles_per_frame.rename(columns={'ANGLE_TURNED': 'MEAN_ANGLE_TURNED'}, inplace=True)
    std_deviation_angles_per_frame = df.groupby('POSITION_T')['ANGLE_TURNED'].std().reset_index()
    std_deviation_angles_per_frame.rename(columns={'ANGLE_TURNED': 'STD_DEVIATION_angles'}, inplace=True)
    median_angles_per_frame = df.groupby('POSITION_T')['ANGLE_TURNED'].median().reset_index()
    median_angles_per_frame.rename(columns={'ANGLE_TURNED': 'MEDIAN_ANGLE_TURNED'}, inplace=True)
    merge = pd.merge(min_angle_per_frame, max_angle_per_frame, on='POSITION_T')
    merge = pd.merge(merge, mean_angles_per_frame, on='POSITION_T')
    merge = pd.merge(merge, std_deviation_angles_per_frame, on='POSITION_T')
    merged = pd.merge(merge, median_angles_per_frame, on='POSITION_T')
    return pd.DataFrame(merged)

def calculate_number_of_frames_per_cell(spot_stats_df):
    # Count the number of frames for each TRACK_ID in the Spot_stats2_df
    frames_per_track = (spot_stats_df.groupby("TRACK_ID").size().reset_index(name="NUM_FRAMES"))

    return frames_per_track

def radial_gradient(radius, fade_color):
    size = 2 * radius
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    mask = np.clip(1 - distance, 0, 1)  # Fade out to edges
    
    return mask * fade_color

butter_df = butter(df, unneccessary_float_columns)
distances_for_each_cell_per_frame_df = calculate_traveled_distances_for_each_cell_per_frame(butter_df) # Call the funciton to clalculate distances for each cell per frame and create the Spot_statistics .csv file
total_distances_df = calculate_total_distance_traveled_for_each_cell(distances_for_each_cell_per_frame_df) # Calling function to calculate the total distance traveled for each cell from the distances_for_each_cell_per_frame_df
net_distances_df = calculate_net_distance_traveled_for_each_cell(distances_for_each_cell_per_frame_df) # Calling function to calculate the net distance traveled for each cell from the distances_for_each_cell_per_frame_df
Track_stats1_df = pd.merge(total_distances_df, net_distances_df, on='TRACK_ID', how='outer') # Merge the total_distances_df and the net_distances_df DataFrames into a new DataFrame: Track_stats1_df
calculate_confinement_ratio_for_each_cell_and_aggregate_with_total_distances_net_distances(Track_stats1_df) # Call the function to calculate confinement ratios from the Track_statistics1_df and write it into the Track_statistics1_df

direction_for_each_cell_per_frame_df = calculate_direction_of_travel_for_each_cell_per_frame(butter_df) # Call the function to calculate direction_for_each_cell_per_frame_df
Spot_stats1_df = pd.merge(distances_for_each_cell_per_frame_df, direction_for_each_cell_per_frame_df, on=['TRACK_ID','POSITION_T'], how='outer') # Merging total_distances_df and direction_for_each_cell_per_frame_df into Spot_stats_df

turn_angles_for_each_cell_per_frame = calculate_turn_angles_for_each_cell_per_frame(Spot_stats1_df) # Call function to calculate the turn angle for each cell per frame
Spot_stats2_df = pd.merge(Spot_stats1_df, turn_angles_for_each_cell_per_frame, on=['TRACK_ID','POSITION_T'], how='outer') # Merging Spot_stats1_df and turn_angles_for_each_cell_per_frame into Spot_stats2_df DataFrame
Spot_stats2_df.to_csv(op.join(save_path, 'Spot_stats.csv')) # Saving the Spot_stats2_df DataFrame into a newly created .csv file

directions_per_cell_df = calculate_directions_per_cell(direction_for_each_cell_per_frame_df) # Call the function to calculate directions_per_cell_df
Track_stats2_df = pd.merge(Track_stats1_df, directions_per_cell_df, on='TRACK_ID', how='outer') # Merge the Track_stats2_df and the directions_per_cell_df into a new DataFrame: Track_stats3_df
frames_per_track = calculate_number_of_frames_per_cell(Spot_stats2_df)
Track_stats3_df = pd.merge(Track_stats2_df, frames_per_track, on="TRACK_ID", how="outer")
Track_stats3_df.to_csv((op.join(save_path, 'Track_stats.csv')), index=False) # Save the Track_stats created DataFrame into a newly created Track_stats_debugging.csv file

distances_per_frame_df = calculate_distances_per_frame(distances_for_each_cell_per_frame_df) # Call the function to calculate distances_per_frame_df
absolute_directions_per_frame_df = calculate_absolute_directions_per_frame(direction_for_each_cell_per_frame_df) # Call the function to calculate directions_per_frame_df
weighted_directions_per_frame = calculate_weighted_directions_per_frame(Spot_stats1_df) # Call the function tp calculate weighted_directions_per_frame
angles_per_frame = calculate_angles_per_frame(Spot_stats2_df)
Time_stats1_df = pd.merge(distances_per_frame_df, absolute_directions_per_frame_df, on='POSITION_T', how='outer') # Merge the distances_per_frame_df and the directions_per_frame_df into a new DataFrame: Time_stats1_df
Time_stats2_df = pd.merge(Time_stats1_df, weighted_directions_per_frame, on='POSITION_T', how='outer') # Merging Time_stats1_df and weighted_directions_per_frame
Time_stats3_df = pd.merge(Time_stats2_df, angles_per_frame, on='POSITION_T', how='outer')
Time_stats3_df.to_csv((op.join(save_path, 'Time_stats.csv')), index=False) # Save the Time_stats1_df into a newly created Time_stats_debugging.csv file



"""def split_dataframe_by_percentiles(df, column_name):
    # Get to know the data frames name
    df_name = [name for name, value in globals().items() if value is df][0]

    # Dictionary to store each DataFrame filtered by percentiles
    dataframes_by_percentile = {}

    # Calculate percentiles and filter the DataFrame for each
    for percentile in range(10, 100, 10):
        threshold_value = df[column_name].quantile(percentile / 100)
        filtered_df = df[df[column_name] > threshold_value]

        # Save the DataFrame
        filename = f'{df_name}_with_{column_name}_thresholded_at_{percentile}th_percentile.csv'
        filtered_df.to_csv(op.join(save_path, filename), index=False)

        # Store in dictionary (optional)
        dataframes_by_percentile[f'threshold_at_{percentile}th_percentile'] = filtered_df

    # Accessing a specific DataFrame for, say, the 30th percentile
    df_thresholded_at_10th_percentile = dataframes_by_percentile['threshold_at_10th_percentile'] # 10th
    df_thresholded_at_20th_percentile = dataframes_by_percentile['threshold_at_20th_percentile'] # 20th
    df_thresholded_at_30th_percentile = dataframes_by_percentile['threshold_at_30th_percentile'] # 30th
    df_thresholded_at_40th_percentile = dataframes_by_percentile['threshold_at_40th_percentile'] # 40th
    df_thresholded_at_50th_percentile = dataframes_by_percentile['threshold_at_50th_percentile'] # 50th
    df_thresholded_at_60th_percentile = dataframes_by_percentile['threshold_at_60th_percentile'] # 60th
    df_thresholded_at_70th_percentile = dataframes_by_percentile['threshold_at_70th_percentile'] # 70th
    df_thresholded_at_80th_percentile = dataframes_by_percentile['threshold_at_80th_percentile'] # 80th
    df_thresholded_at_90th_percentile = dataframes_by_percentile['threshold_at_90th_percentile'] # 90th


    return df_thresholded_at_10th_percentile, df_thresholded_at_20th_percentile, df_thresholded_at_30th_percentile, df_thresholded_at_40th_percentile, df_thresholded_at_50th_percentile, df_thresholded_at_60th_percentile, df_thresholded_at_70th_percentile, df_thresholded_at_80th_percentile, df_thresholded_at_90th_percentile
Track_stats_thresholded_at_10th_percentile, Track_stats_thresholded_at_20th_percentile, Track_stats_thresholded_at_30th_percentile, Track_stats_thresholded_at_40th_percentile, Track_stats_thresholded_at_50th_percentile, Track_stats_thresholded_at_60th_percentile, Track_stats_thresholded_at_70th_percentile, Track_stats_thresholded_at_80th_percentile, Track_stats_thresholded_at_90th_percentile = split_dataframe_by_percentiles(Track_stats3_df, 'NET_DISTANCE')"""

# You should try: split_dataframe_by_percentiles(df, column_name); column_name = 'NET_DISTANCE', 'TOTAL_DISTANCE', 'CONFINEMENT_RATIO', 'SPEED_MEDIAN AND OR MEAN, ETC 


"""def histogram_cells_distance(df, metric, str):
    
    # Sort the DataFrame by 'TOTAL_DISTANCE' in ascending order
    df_sorted = df.sort_values(by=metric)

    norm = mcolors.Normalize(vmin=df_sorted["NUM_FRAMES"].min(), vmax=df_sorted["NUM_FRAMES"].max())
    cmap = plt.colormaps["ocean_r"]

    # Create new artificial IDs for sorting purposes (1 for lowest distance, N for highest)
    df_sorted["Artificial_ID"] = range(1, len(df_sorted) + 1)

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_tight_layout(True)
    width = 6

    # Loop through each row to plot each cell's data
    for idx, row in df_sorted.iterrows():
        artificial_id = row["Artificial_ID"]
        total_distance = row[metric]
        num_frames = row["NUM_FRAMES"]

        # Get the color based on the number of frames using the viridis colormap
        line_color = cmap(norm(num_frames))

        # Plot the "chimney" or vertical line
        ax.vlines(
            x=artificial_id,  # X position for the cell
            ymin=total_distance,  # Starting point of the line (y position)
            ymax=total_distance + num_frames,  # End point based on number of frames (height)
            color=line_color,
            linewidth=width,
            )

        ax.hlines(
            y=total_distance,  # Y position for the lavender line (starting distance)
            xmin=artificial_id - (width - 5.850),  # Start a little to the left of the main line
            xmax=artificial_id + (width - 5.795),  # End a little to the right of the main line
            color="lavender",  # Color of the small line
            linewidth=2,  # Thickness of the small line
            zorder=5,  # Ensure it's drawn above other elements
        )

        # Add the mean number of frames as text above each chimney
        ax.text(
        artificial_id,  # X position (same as the chimney)
        total_distance + num_frames + 1,  # Y position (slightly above the chimney)
        f"{round(num_frames)}",  # The text to display (formatted mean)
        ha='center',  # Horizontal alignment center
        va='bottom',  # Vertical alignment bottom
        fontsize=8,  # Adjust font size if necessary
        color='black'  # Color of the text
        )

        x = int(row['Artificial_ID'])

        plt.xticks(range(x), rotation=90) # add loads of ticks
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)

        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2 # inch margin
        s = maxsize/plt.gcf().dpi*x+2*m
        margin = m/plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # Dummy mappable to create the colorbar
    # fig.colorbar(sm, ax=ax, label="Number of Frames")

    max_y = df_sorted[metric].max()
    num_x_values = df_sorted[metric].count()

    # Adjust the plot aesthetics
    plt.xticks(range(num_x_values)) # add loads of ticks
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)

    # Set ticks, labels and title
    ax.set_xticks(range(1, num_x_values + 1))
    ax.set_yticks(np.arange(0, max_y + 1, 10))
    ax.set_xlabel(f"Cells (sorted by {str} distance)")
    ax.set_ylabel(f"{str} distance traveled [microns]")
    ax.set_title(f"{str} Distance Traveled by Cells\nWith Length Representing Number of Frames")

    # Invert x-axis so the highest distance is on the left
    plt.gca().invert_xaxis()

    ax.set_xlim(right=0, left=num_x_values+1)  # Adjust the left limit as needed

    # Show the plot
    plt.savefig(op.join(save_path, f"02f_Histogram_{str}_distance_traveled_per_cell.png"))
    # plt.show()
# histogram_cells_distance(Track_stats3_df, 'NET_DISTANCE', 'Net')
# histogram_cells_distance(Track_stats3_df, 'TOTAL_DISTANCE', 'Total')"""

"""def histogram_nth_percentile_distance(df, metric, num_groups, percentiles, str, threshold):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    # Sort the DataFrame by 'NET_DISTANCE' in ascending order
    df_sorted = df.sort_values(by=metric)

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_tight_layout(True)
    width = 6

    # Normalize the 'NUM_FRAMES' column for color mapping
    norm = mcolors.Normalize(vmin=df_sorted["NUM_FRAMES"].min(), vmax=df_sorted["NUM_FRAMES"].max())
    cmap = plt.colormaps["ocean_r"]

    # Number of groups (chimneys) and size of each group (5% each)
    group_size = len(df_sorted) // num_groups

    # Loop over each group and plot the aggregate statistics
    for i in range(num_groups):
        # Define group indices
        group_start = i * group_size
        group_end = (i + 1) * group_size if i != num_groups - 1 else len(df_sorted)

        # Get the current group data
        group_data = df_sorted.iloc[group_start:group_end]

        # Calculate the aggregate statistics for the group
        group_mean_distance = group_data[metric].mean()
        group_mean_frames = group_data["NUM_FRAMES"].mean()

        # Set x-position of the group (e.g., center it based on group index)
        group_x = i + 1

        # Get the color based on the average number of frames in the group
        line_color = cmap(norm(group_mean_frames))

        # Plot the "chimney" representing the group
        ax.vlines(
            x=group_x,  # X position for the group
            ymin=group_mean_distance,  # Starting point of the line (y position)
            ymax=group_mean_distance + group_mean_frames,  # End point based on average number of frames (height)
            color=line_color,
            linewidth=width,
            )

        # Optionally, add a horizontal "base" line to show the start of the group
        ax.hlines(
            y=group_mean_distance,  # Y position for the base line (starting distance)
            xmin=group_x - (width - 5.850),  # Start a little to the left of the main line
            xmax=group_x + (width - 5.795),  # End a little to the right of the main line
            color="lavender",  # Color of the small line
            linewidth=2,  # Thickness of the small line
            zorder=5,  # Ensure it's drawn above other elements
            )

        # Add the mean number of frames as text above each chimney
        ax.text(
            group_x,  # X position (same as the chimney)
            group_mean_distance + group_mean_frames + 1,  # Y position (slightly above the chimney)
            f"{round(group_mean_frames)}",  # The text to display (formatted mean)
            ha='center',  # Horizontal alignment center
            va='bottom',  # Vertical alignment bottom
            fontsize=8,  # Adjust font size if necessary
            color='black'  # Color of the text
            )

        x = group_x

        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2 # inch margin
        s = maxsize/plt.gcf().dpi*x+2*m
        margin = m/plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    max_y = df_sorted[metric].max() 

    # Adjust the plot aesthetics
    plt.xticks(range(group_x)) # add loads of ticks
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)

    ax.set_xticks(range(1, num_groups + 1))
    ax.set_xticklabels([f"{percentiles * (i + 1)}%" for i in range(num_groups)], rotation=90)
    ax.set_yticks(np.arange(0, max_y + 1, 10))
    ax.set_xlabel(f"Cell groups (each {percentiles}% of sorted data)")
    ax.set_ylabel(f"{str} distance traveled [microns]")
    ax.set_title(f"{str} Distance Traveled by Cells (Grouped by Percentile)\nWith Length Representing Average Number of Frames")
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)

    # Invert x-axis so the highest distance is on the left
    ax.invert_xaxis()

    ax.set_xlim(right=0, left=num_groups+1)  # Adjust the left limit as needed

    # Show the plot
    plt.savefig((op.join(save_path, f"02f_Histogram_{str}_distance_traveled_{percentiles}th_percentiles{threshold}.png")))
    # plt.show()
histogram_nth_percentile_distance(Track_stats3_df, 'TOTAL_DISTANCE', 20, 5, 'Total', None)
histogram_nth_percentile_distance(Track_stats3_df, 'TOTAL_DISTANCE', 100, 1, 'Total', None)
histogram_nth_percentile_distance(Track_stats3_df, 'NET_DISTANCE', 20, 5, 'Net', None)
histogram_nth_percentile_distance(Track_stats3_df, 'NET_DISTANCE', 100, 1, 'Net', None)
# histogram_nth_percentile_distance(Track_stats_thresholded_at_20th_percentile, 'TOTAL_DISTANCE', 20, 5, 'Total', 'thresholded_at_20th_percentile') # 20th
# histogram_nth_percentile_distance(Track_stats_thresholded_at_20th_percentile, 'NET_DISTANCE', 20, 5, 'Net', 'thresholded_at_20th_percentile')
# histogram_nth_percentile_distance(Track_stats_thresholded_at_40th_percentile, 'TOTAL_DISTANCE', 20, 5, 'Total', 'thresholded_at_40th_percentile') # 40th
# histogram_nth_percentile_distance(Track_stats_thresholded_at_40th_percentile, 'NET_DISTANCE', 20, 5, 'Net', 'thresholded_at_40th_percentile')
# histogram_nth_percentile_distance(Track_stats_thresholded_at_60th_percentile, 'TOTAL_DISTANCE', 20, 5, 'Total', 'thresholded_at_60th_percentile') # 60th
# histogram_nth_percentile_distance(Track_stats_thresholded_at_60th_percentile, 'NET_DISTANCE', 20, 5, 'Net', 'thresholded_at_60th_percentile')
# histogram_nth_percentile_distance(Track_stats_thresholded_at_80th_percentile, 'TOTAL_DISTANCE', 20, 5, 'Total', 'thresholded_at_80th_percentile') # 80th
# histogram_nth_percentile_distance(Track_stats_thresholded_at_80th_percentile, 'NET_DISTANCE', 20, 5, 'Net', 'thresholded_at_80th_percentile')"""

"""def donut(df, ax, outer_radius, inner_radius, kde_bw):
    # Extend the data circularly to account for wrap-around at 0 and 2*pi
    extended_data = np.concatenate([df - 2 * np.pi, df, df + 2 * np.pi])

    # Create a grid of theta values (angles)
    theta_grid = np.linspace(0, 2 * np.pi, 360)  # 360 points over full circle
    
    # Create a grid of radii
    r_grid = np.linspace(inner_radius, outer_radius, 100)  # Radius from inner to outer edge
    
    # Compute KDE values for the extended data
    kde = gaussian_kde(extended_data, bw_method=kde_bw)
    kde_values = kde.evaluate(theta_grid)  # Evaluate KDE on the regular theta grid
    
    # Repeat KDE values across radii to create the heatmap data
    kde_values = np.tile(kde_values, (r_grid.size, 1))
    
    # Normalize KDE values for consistent color mapping
    norm = Normalize(vmin=kde_values.min(), vmax=kde_values.max())
    
    # Create the meshgrid for the polar plot
    theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
    
    # Remove polar grid lines and labels
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)  # Hide the outer frame

    return theta_mesh, r_mesh, kde_values, norm"""

"""def df_gaussian_donut(df, metric, subject, heatmap, threshold):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    diameter=2
    width_ratio=0.3
    kde_bw=0.1

    df=df[metric]

    # Calculate radius and width from the diameter
    outer_radius = diameter / 2
    width = width_ratio * outer_radius
    inner_radius = outer_radius - width
    
    theta_mesh, r_mesh, kde_values, norm = donut(df, ax, outer_radius, inner_radius, kde_bw)
    
    # Set title and figure text
    ax.set_title(f'Heatmap of Migration Direction ({subject})', pad=20, ha='center', fontsize=title_size)
    plt.figtext(0.515, 0.01, 'weighted by confinement ratio', ha='center', color=figtext_color, fontsize=figtext_size)
    
    # Add a colorbar
    cbar = plt.colorbar(ax.pcolormesh(theta_mesh, r_mesh, kde_values, shading='gouraud', cmap=heatmap, norm=norm), ax=ax, fraction=0.04, orientation='horizontal', pad=0.1)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)  # Remove outline
    
    # Add min and max labels below the colorbar
    cbar.ax.text(0.05, -0.4, 'min', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.95, -0.4, 'max', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)

    # Add the density label below the min and max labels
    cbar.set_label('Density', labelpad=10, fontsize=label_size)
    
    plt.savefig(op.join(save_path, f'04a_Plot_donut_heatmap-migration_direction_{subject}{threshold}.png'), dpi=300)
    # plt.show()
df_gaussian_donut(Track_stats3_df, 'MEAN_DIRECTION_RAD', 'Cells', 'inferno', None)
# df_gaussian_donut(Track_stats_thresholded_at_20th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'inferno', 'thresholded_at_20th_percentile') # 20th
df_gaussian_donut(Track_stats_thresholded_at_40th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'inferno', 'thresholded_at_40th_percentile') # 40th
df_gaussian_donut(Track_stats_thresholded_at_60th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'inferno', 'thresholded_at_60th_percentile') # 60th
df_gaussian_donut(Track_stats_thresholded_at_80th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'inferno', 'thresholded_at_80th_percentile') # 80th
df_gaussian_donut(Track_stats_thresholded_at_90th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'inferno', 'thresholded_at_90th_percentile') # 90th
df_gaussian_donut(Time_stats3_df, 'MEAN_DIRECTION_RAD_weight', 'Frames', 'viridis', None)"""

"""def ticks_for_mean_direction_clock(df, metric, subject, threshold):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    df=df[metric]

    # Calculate radius and width from the diameter
    radius_full = 1
    length_from_outer_border = 1.6
    
    # Create theta values (angles) for each data point
    theta = df % (2 * np.pi)  # Ensure the angles are within [0, 2π)
    
    # Plot the lines for the donut shape
    for angle in theta:
        # Create a line segment from inner to outer radius
        line_theta = [angle, angle]
        line_r = [radius_full, length_from_outer_border]
        ax.plot(line_theta, line_r, color='black', linewidth=0.5)  # Adjust linewidth as needed

    # Remove polar grid lines and labels
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)  # Hide the outer frame
    
    # Set title and figure text
    ax.set_title(f'Mean Direction of Travel per Each {subject}', pad=20, ha='center')
    plt.figtext(0.5, 0.01, 'weighted by confinement ratio', color='black', ha='center', fontsize=10)
    
    plt.savefig(op.join(save_path, f'clock_mean_direction_per_{subject}{threshold}.png'), dpi=300)
    # plt.show()
# ticks_for_mean_direction_clock(Track_stats3_df, 'MEAN_DIRECTION_RAD', 'Cell', None)
# ticks_for_mean_direction_clock(Track_stats_thresholded_at_20th_percentile, 'MEAN_DIRECTION_RAD', 'Cell', 'thresholded_at_20th_percentile') # 20th
# ticks_for_mean_direction_clock(Track_stats_thresholded_at_40th_percentile, 'MEAN_DIRECTION_RAD', 'Cell', 'thresholded_at_40th_percentile') # 40th
# ticks_for_mean_direction_clock(Track_stats_thresholded_at_60th_percentile, 'MEAN_DIRECTION_RAD', 'Cell', 'thresholded_at_60th_percentile') # 60th
# ticks_for_mean_direction_clock(Track_stats_thresholded_at_80th_percentile, 'MEAN_DIRECTION_RAD', 'Cell', 'thresholded_at_80th_percentile') # 80th
# ticks_for_mean_direction_clock(Time_stats3_df, 'MEAN_DIRECTION_RAD_weight', 'Frame', None)"""

"""def combined_plot(df, metric, subject, colormap, threshold):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    df = df[metric]

    def donut_plot(ax, df):
        # DONUT PLOT
        outer_radius = 1.3
        inner_radius = 1.0
        kde_bw = 0.1

        theta_mesh, r_mesh, kde_values, norm = donut(df, ax, outer_radius, inner_radius, kde_bw)

        # Plot the KDE heatmap
        c = ax.pcolormesh(theta_mesh, r_mesh, kde_values, cmap=colormap, shading='gouraud', norm=norm, alpha=0.7)

        return c

    def clock_plot(ax, df):
        # CLOCK PLOT
        outer_radius = 0.9
        inner_radius = 0.75

        # Create theta values (angles) for each data point
        theta = df % (2 * np.pi)

        # Plot the lines for the clock plot
        for angle in theta:
            line_theta = [angle, angle]
            line_r = [inner_radius, outer_radius]
            ax.plot(line_theta, line_r, color='black', linewidth=0.5)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

    def kde_plot(ax, df):
        # KDE PLOT
        x = np.cos(df)
        y = np.sin(df)

        kde = gaussian_kde([x, y])

        # Define the grid for evaluation
        theta = np.linspace(0, 2 * np.pi, 360)
        x_grid = np.cos(theta)
        y_grid = np.sin(theta)

        z = kde.evaluate([x_grid, y_grid])

        # Normalize the z values to fit within the grid limits
        norm_z = z / z.max()  # Scaling to fit within the radial limit set by ax1.set_ylim(0, 0.5)

        z = norm_z * 0.25

        # Plot the KDE
        ax.plot(theta, z, label='Circular KDE', color='None', zorder=3)
        ax.fill(theta, z, alpha=0.8, color='lightslategrey', zorder=3)
        
        # Calculate the mean direction
        mean_direction = np.arctan2(np.mean(y), np.mean(x))
        ax.plot([mean_direction, mean_direction], [0, max(z)], linestyle='--', color='black', alpha=0.63, linewidth=1.5, zorder=4) # Plot the dashed line in the mean direction
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

    def make_grid(ax):
        radius = 0.6

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

        # Create angular grid lines
        def for_angular_lines(theta, color, linestyle, lw):
            for angle in theta:
                ax.plot([angle, angle], [0, (radius + 0.05)], color=color, linestyle=linestyle, linewidth=lw, zorder=0)

        theta1 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.deg2rad(45)
        theta2 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        theta3 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.deg2rad(22.5)
        color1 = 'darkgrey'
        color2 = 'darkgrey'
        color3 = 'lightgrey'
        linestyle1 = '-'
        linestyle2 = '--'
        linestyle3 = '--'
        lw1 = 0.6
        lw2 = 0.5
        lw3 = 0.5
        for_angular_lines(theta1, color1, linestyle1, lw1)
        for_angular_lines(theta2, color2, linestyle2, lw2)
        for_angular_lines(theta3, color3, linestyle3, lw3)

        # Create radial grid lines
        r = np.linspace(0, radius, 5)
        for r_val in r:
            ax.plot(np.linspace(0, 2 * np.pi, 360), [r_val]*360, color='lightgrey', linestyle='--', linewidth=0.5, zorder=0)


    # Create the figure and polar axis
    fig = plt.figure(figsize=(8, 8))
    ax3 = fig.add_subplot(111, projection='polar')
    ax1 = fig.add_subplot(111, projection='polar', frame_on=False)  # KDE plot with grid
    ax2 = fig.add_subplot(111, projection='polar', frame_on=False)  # Clock and Donut plots

    # Set radial limits to ensure proper layering
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 1.3)
    ax3.set_ylim(0, 1.3)

    make_grid(ax3)

    # Plot the KDE plot with grid
    kde_plot(ax1, df)
    
    # Plot the clock and donut plots on top
    donut_plot(ax2, df)
    clock_plot(ax2, df)
    plt.savefig(op.join(save_path, f'06a_Plot_combined_{subject}{threshold}.png'), dpi=300)
    # plt.show()
# combined_plot(Track_stats3_df, 'MEAN_DIRECTION_RAD', 'Cells', 'viridis', None)
# combined_plot(Track_stats_thresholded_at_20th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'viridis', 'thresholded_at_20th_percentile') # 20th
# combined_plot(Track_stats_thresholded_at_40th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'viridis', 'thresholded_at_40th_percentile') # 40th
# combined_plot(Track_stats_thresholded_at_60th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'viridis', 'thresholded_at_60th_percentile') # 60th
# combined_plot(Track_stats_thresholded_at_80th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'viridis', 'thresholded_at_80th_percentile') # 80th
# combined_plot(Time_stats3_df, 'MEAN_DIRECTION_RAD_weight', 'Frames', 'inferno', None)"""

def track_visuals(df, df2):
    # fig and ax definition
    fig, ax = plt.subplots(figsize=(13, 10))

    # Ensuring that dataframse have required data
    track_ids = df2['TRACK_ID'].unique()

    # Filter df2 to only include rows where TRACK_ID is in df's track_ids
    df_filtered = df[df['TRACK_ID'].isin(track_ids)]

    net_distances = df2[['TRACK_ID', 'NET_DISTANCE']]

    # Normalize the NET_DISTANCE to a 0-1 range
    dmin = net_distances['NET_DISTANCE'].min()
    dmax = net_distances['NET_DISTANCE'].max()
    norm = plt.Normalize(vmin=dmin, vmax=dmax)
    colormap = plt.cm.jet

    # Create a dictionary to store the color for each track based on its confinement ratio
    track_colors = {}
    for track_id in track_ids:
        ratio = net_distances[net_distances['TRACK_ID'] == track_id]['NET_DISTANCE'].values[0]
        track_colors[track_id] = colormap(norm(ratio))

    # Set up the plot limits
    x_min, x_max = df_filtered['POSITION_X'].min(), df_filtered['POSITION_X'].max()
    y_min, y_max = df_filtered['POSITION_Y'].min(), df_filtered['POSITION_Y'].max()
    ax.set_aspect('1', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Position X [microns]')
    ax.set_ylabel('Position Y [microns]')
    ax.set_title('Track Visualization', fontsize=title_size)
    ax.set_facecolor('gainsboro')
    ax.grid(True, which='both', axis='both', color='whitesmoke', linewidth=0.5)

    # Manually set the major tick locations and labels
    x_ticks_major = np.arange(x_min, x_max, 200)  # Adjust the step size as needed
    y_ticks_major = np.arange(y_min, y_max, 200)  # Adjust the step size as needed
    ax.set_xticks(x_ticks_major)
    ax.set_yticks(y_ticks_major)
    ax.set_xticklabels([f'{tick:.0f}' for tick in x_ticks_major])
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks_major])

    # Enable minor ticks and set their locations
    ax.minorticks_on()
    x_ticks_minor = np.arange(x_min, x_max, 50)  # Minor ticks every 50 microns
    y_ticks_minor = np.arange(y_min, y_max, 50)  # Minor ticks every 50 microns
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.set_yticks(y_ticks_minor, minor=True)
    

    # Access and modify tick labels
    # for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    #      label.set_fontproperties(roboto_ticks)
    ax.tick_params(axis='both', which='major', labelsize=8)

    return fig, ax, track_ids, track_colors, norm, colormap

def visualize_full_tracks(df, df2, threshold):  #Trakcs visualisation

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    # Using the  track_visuals function
    fig, ax, track_ids, track_colors, norm, colormap = track_visuals(df, df2)

    # Plot the full tracks
    for track_id in track_ids:
        track_data = df[df['TRACK_ID'] == track_id]
        x = track_data['POSITION_X']
        y = track_data['POSITION_Y']
        ax.plot(x, y, lw=1, color=track_colors[track_id], label=f'Track {track_id}')
        
        if len(x) > 1:
            # Add arrow to indicate direction
            dx = x.diff().iloc[-1]
            dy = y.diff().iloc[-1]
            if dx != 0 or dy != 0:
                endpoint = Circle(
                    (x.iloc[-1], y.iloc[-1]),  # Center of the circle at the last point
                    radius=5,
                                    # Set the radius to your preferred size
                    color=track_colors[track_id],
                    fill=True                   # Set to False if you want only the outline
                    )
                ax.add_patch(endpoint)

    plt.savefig((op.join(save_path, f'01a_Full_tracks_snapshot{threshold}.png')))
    plt.show()
visualize_full_tracks(df, Track_stats2_df, None)
# visualize_full_tracks(df, Track_stats_thresholded_at_20th_percentile, 'thresholded_at_20th_percentile') # 20th
# visualize_full_tracks(df, Track_stats_thresholded_at_40th_percentile, 'thresholded_at_40th_percentile') # 40th
# visualize_full_tracks(df, Track_stats_thresholded_at_60th_percentile, 'thresholded_at_60th_percentile') # 60th
# visualize_full_tracks(df, Track_stats_thresholded_at_80th_percentile, 'thresholded_at_80th_percentile') # 80th
# visualize_full_tracks(df, Track_stats_thresholded_at_90th_percentile, 'thresholded_at_90th_percentile') # 90th

# def animate_tracks_growth_over_time(df, df2, threshold):    # Animated tracks visualization

#     # Recognizing the presence of a threshold
#     if threshold == None:
#         threshold = '_no_threshold'
#     else:
#         threshold = '_' + threshold

#     fig, ax, track_ids, track_colors, norm, colormap = track_visuals(df, df2)
    
#     # Create a dictionary to store the line objects and arrows for each track
#     track_lines = {track_id: ax.plot([], [], lw=1, color=track_colors[track_id], label=f'Track {track_id}')[0] for track_id in track_ids}
#     arrows = {track_id: [] for track_id in track_ids}

#     # Create a smaller colorbar on the left side of the plot (centered vertically)
#     cbar_width = 0.002  # Width of the colorbar
#     cbar_x_pos = 0.02  # X position for the colorbar (left side of the plot)
#     cbar_height = 0.2  # Height of the colorbar (adjust as needed)
#     cbar_y_pos = 0.1 + (0.8 - cbar_height) / 2  # Vertically center the colorbar

#     cbar_ax = fig.add_axes([cbar_x_pos, cbar_y_pos, cbar_width, cbar_height])  # [left, bottom, width, height]
#     cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax)
#     cbar.set_label('net distance', rotation=270, labelpad=17)
#     cbar.ax.yaxis.set_label_position('left')
#     cbar.ax.tick_params(labelsize=8)
#     cbar.outline.set_visible(False)  # Hide the colorbar outline

#     # Create a new axis for displaying the frame count
#     frame_ax = fig.add_axes([0.1, 0.9, 0.8, 0.05])
#     frame_ax.axis('off')  # Hide the new axis
#     frame_text = frame_ax.text(0.5, 0.5, '', ha='center', va='center', transform=frame_ax.transAxes, fontsize=8, )
    
#     def init():
#         for line in track_lines.values():
#             line.set_data([], [])
#         frame_text.set_text('')
#         return list(track_lines.values()) + [frame_text]
    
#     def update(frame):
#         # Filter data up to the current frame
#         unique_times = sorted(df['POSITION_T'].unique())
#         current_time = unique_times[min(frame, len(unique_times) - 1)]
#         current_data = df[df['POSITION_T'] <= current_time]
        
#         # Update line data and arrows for each track
#         for track_id in track_ids:
#             track_data = current_data[current_data['TRACK_ID'] == track_id]
#             # Sort the track data by time to ensure proper sequence
#             track_data = track_data.sort_values(by='POSITION_T')
#             x = track_data['POSITION_X']
#             y = track_data['POSITION_Y']
#             track_lines[track_id].set_data(x, y)
            
#             # Clear previous arrows
#             for arrow in arrows[track_id]:
#                 arrow.remove()
#             arrows[track_id] = []
            
#             if len(x) > 1:
#                 # Calculate differences to place the arrows
#                 dx = x.diff().iloc[-1]
#                 dy = y.diff().iloc[-1]
#                 if dx != 0 or dy != 0:
#                     # Create the arrow
#                     arrow = FancyArrowPatch((x.iloc[-2], y.iloc[-2]),
#                                             (x.iloc[-1] + dx, y.iloc[-1] + dy),
#                                             color=track_colors[track_id],
#                                             arrowstyle='->', mutation_scale=5)
#                     ax.add_patch(arrow)
#                     arrows[track_id].append(arrow)
        
#         # Update the frame count text
#         frame_text.set_text(f'Frame: {frame + 1}/{num_frames}')

#         return list(track_lines.values()) + [frame_text] + [arrow for arrow_list in arrows.values() for arrow in arrow_list]
    
#     # Create the animation object
#     num_frames = len(df['POSITION_T'].unique())
#     ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=1, repeat=False)
#     ani.save(op.join(save_path, f'01b_Tracks_animation{threshold}.gif'), writer='pillow', fps=15)
#     # plt.show()
# animate_tracks_growth_over_time(df, Track_stats2_df, None)
# animate_tracks_growth_over_time(df, Track_stats_thresholded_at_20th_percentile, 'thresholded_at_20th_percentile') # 20th
# animate_tracks_growth_over_time(df, Track_stats_thresholded_at_40th_percentile, 'thresholded_at_40th_percentile') # 40th
# animate_tracks_growth_over_time(df, Track_stats_thresholded_at_60th_percentile, 'thresholded_at_60th_percentile') # 60th
# animate_tracks_growth_over_time(df, Track_stats_thresholded_at_80th_percentile, 'thresholded_at_80th_percentile') # 80th

cmap_cells = mcolors.LinearSegmentedColormap.from_list("", ["#9b598910", "#9b181eff"]) #303030
cmap_frames = plt.get_cmap('viridis')

"""def migration_directions_with_kde_plus_mean(df, metric, subject, scaling_metric, cmap_normalization_metric, cmap, threshold):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold

    df_mean_direction = df[metric]

    # Prepare for KDE plot
    x_kde = np.cos(df_mean_direction)
    y_kde = np.sin(df_mean_direction)
    kde = gaussian_kde([x_kde, y_kde])

    # Define the grid for evaluation
    theta_kde = np.linspace(0, 2 * np.pi, 360)
    x_grid = np.cos(theta_kde)
    y_grid = np.sin(theta_kde)

    # Evaluate the KDE on the grid and normalize
    z_kde = kde.evaluate([x_grid, y_grid])
    z_kde = z_kde / z_kde.max() * 0.5  # Normalize to fit within the radial limit

    # Calculate the mean direction
    mean_direction = np.arctan2(np.mean(y_kde), np.mean(x_kde))

    # Start plotting
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})

    # Plot KDE
    ax.plot(theta_kde, z_kde, label='Circular KDE', color='None', zorder=5)
    ax.fill(theta_kde, z_kde, alpha=0.25, color='#1b5a9e', zorder=5)

    # Directional Arrows
    scaling_max = df[scaling_metric].max()

    # Normalization of the color map
    if cmap_normalization_metric == None:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        normalization_min = df[cmap_normalization_metric].min()
        normalization_max = df[cmap_normalization_metric].max()
        norm = mcolors.Normalize(vmin=normalization_min, vmax=normalization_max)

    # Row itteration
    for _, row in df.iterrows():
        scaling_metrics = row[scaling_metric]
        mean_direction_rad = row[metric]
        arrow_length = scaling_metrics / scaling_max

        if cmap_normalization_metric == None:
            color = cmap(norm(arrow_length))
        else:
            color = cmap(norm(row[cmap_normalization_metric]))

        if arrow_length == 0:
            continue  # Skip if the arrow length is zero

        # Dynamically adjust the head size based on arrow_length
        scaling_factor = 1 / arrow_length if arrow_length != 0 else 1
        head_width = 0.011 * scaling_factor
        head_length = 0.013

        ax.arrow(mean_direction_rad, 0, 0, arrow_length, color=color, linewidth=0.75, 
                head_width=head_width, head_length=head_length, zorder=4)

    # Plot the dashed line in the mean direction
    ax.plot([mean_direction, mean_direction], [0, 1], linestyle='--', color='darkslateblue', alpha=0.93, linewidth=2.5, zorder=6)

    # Hide the polar plot frame (spines) but keep the grid visible
    ax.spines['polar'].set_visible(False)
    # Customize grid lines (if needed)
    ax.grid(True, 'major', color='#C6C6C6', linestyle='-', linewidth=0.5, zorder=0)

    # Access and customize the radial grid lines
    radial_lines = ax.get_xgridlines()
    for i, line in enumerate(radial_lines):
        if i % 2 == 0:  # Customize every other radial grid line
            line.set_linestyle('--')
            line.set_color('#E6E6E6')
            line.set_linewidth(0.5)

    radial_lines = ax.get_ygridlines()
    for i, line in enumerate(radial_lines):
        line.set_linestyle('--')
        line.set_color('#E6E6E6')
        line.set_linewidth(0.5)

    # Customize the appearance of the polar plot
    ax.set_title(f'Mean Direction of Travel with Kernel Density Estimate\n$\it{{{subject}}}$', fontsize=title_size)
    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticklabels([])  # Remove angular labels

    # Save the plot
    plt.savefig(op.join(save_path, f'02c_Plot_directions_of_travel_with_mean_and_kernel_density_estimate_{subject}{threshold}.png'), dpi=500)
    # plt.show()
migration_directions_with_kde_plus_mean(Track_stats3_df, 'MEAN_DIRECTION_RAD', 'Cells', 'CONFINEMENT_RATIO', None, cmap_cells, None)
# migration_directions_with_kde_plus_mean(Track_stats_thresholded_at_20th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'CONFINEMENT_RATIO', None, cmap_cells, 'thresholded_at_20th_percentile') # 20th
migration_directions_with_kde_plus_mean(Track_stats_thresholded_at_40th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'CONFINEMENT_RATIO', None, cmap_cells, 'thresholded_at_40th_percentile') # 40th
migration_directions_with_kde_plus_mean(Track_stats_thresholded_at_60th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'CONFINEMENT_RATIO', None, cmap_cells, 'thresholded_at_60th_percentile') # 60th
migration_directions_with_kde_plus_mean(Track_stats_thresholded_at_80th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'CONFINEMENT_RATIO', None, cmap_cells, 'thresholded_at_80th_percentile') # 80th
migration_directions_with_kde_plus_mean(Track_stats_thresholded_at_90th_percentile, 'MEAN_DIRECTION_RAD', 'Cells', 'CONFINEMENT_RATIO', None, cmap_cells, 'thresholded_at_90th_percentile') # 90th
migration_directions_with_kde_plus_mean(Time_stats3_df, 'MEAN_DIRECTION_RAD_weight', 'Frames_abs', 'MEAN_DISTANCE', 'POSITION_T', cmap_frames, None)
migration_directions_with_kde_plus_mean(Time_stats3_df, 'MEAN_DIRECTION_RAD_abs', 'Frames_weight', 'MEAN_DISTANCE', 'POSITION_T', cmap_frames, None)"""


"""# STATISTICS

def calculate_direction_of_travel_for_each_cell_return_txt_file(df, txt_file): # look into calculations

    # Initialize counters and weighted sums
    north = 0
    south = 0
    west = 0
    east = 0

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        mean_direction_deg = row['MEAN_DIRECTION_DEG']

        # Determine if the direction is North or South based on the direction
        if 45 < mean_direction_deg < 135:
            east += 1
        elif 135 < mean_direction_deg < 225:
            south += 1
        elif 225 < mean_direction_deg < 315:
            west += 1
        else:
            north += 1

    total = north + south + east + west
    horizontal = east + west

    # Handle the case where there are no directions
    if total == 0:
        print("No direction data available.")
        return

    # Calculate weighted percentages
    north_percentage = (north / total) * 100
    south_percentage = (south / total) * 100
    east_percentage = (east / total) * 100
    west_percentage = (west / total) * 100
    horizontal_percentage = (horizontal / total) * 100

    print(
        f'Cells traveling North: {north_percentage:.2f} % ({north})'
        f'\nCells traveling South: {south_percentage:.2f} % ({south})'
        f'\nCells traveling horizontally: {horizontal_percentage:.2f} % ({horizontal}) [travelling west: {west_percentage:.2f} ({west}); travelling east: {east_percentage:.2f} ({east})]'
        )

    # Save the results as a text file
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(
            f'Cells traveling North: {north_percentage:.2f} % ({north})\n'
            f'Cells traveling South: {south_percentage:.2f} % ({south})\n'
            f'Cells traveling horizontally: {horizontal_percentage:.2f} % ({horizontal}) [travelling west: {west_percentage:.2f} ({west}); travelling east: {east_percentage:.2f} ({east})]\n\n'
            )

def rayleigh_test(df, txt_file):

    df = df['MEAN_DIRECTION_RAD']

    def calculate_rayleigh_test(df):
        # Tests if given directions are uniformly distributed using Rayleigh test.
        # Calculate the Rayleigh test statistic
        r = np.mean(np.exp(1j * df))

        # Calculate the Rayleigh test statistic
        z = np.sqrt(len(df) * r.real**2 + len(df) * r.imag**2)

        # Calculate the p-value
        p_value = rayleigh.sf(z)

        # Set significance level
        alpha = 0.05

        return p_value, alpha

    p_value, alpha = calculate_rayleigh_test(df)

    # Save the Rayleigh test results to the same file
    with open(txt_file, 'a', encoding='utf-8') as file:
        print('')
        if p_value > alpha:
            print(f'Data is uniformly distributed.  P-value: {p_value:.3e} is greater than {alpha}')
            file.write(f'Data is uniformly distributed. P-value: {p_value:.3e} is greater than {alpha}\n')
        else:
            print(f'Data is not uniformly distributed.  P-value: {p_value:.3e} is smaller than {alpha}')
            file.write(f'Data is not uniformly distributed.  P-value: {p_value:.3e} is smaller than {alpha}\n')
    
def statistics(df, threshold):

    # Recognizing the presence of a threshold
    if threshold == None:
        threshold = '_no_threshold'
    else:
        threshold = '_' + threshold


    txt_file = op.join(save_path, f'Rayleigh_test_and_Direction_of_migration{threshold}_results.txt')

    calculate_direction_of_travel_for_each_cell_return_txt_file(df, txt_file)
    rayleigh_test(df, txt_file)
    print('\n')

statistics(Track_stats3_df, None)
statistics(Track_stats_thresholded_at_20th_percentile, 'thresholded_at_20th_percentile') # 20th
statistics(Track_stats_thresholded_at_40th_percentile, 'thresholded_at_40th_percentile') # 40th
statistics(Track_stats_thresholded_at_60th_percentile, 'thresholded_at_60th_percentile') # 60th
statistics(Track_stats_thresholded_at_80th_percentile, 'thresholded_at_80th_percentile') # 80th
statistics(Track_stats_thresholded_at_90th_percentile, 'thresholded_at_90th_percentile') # 90th
"""