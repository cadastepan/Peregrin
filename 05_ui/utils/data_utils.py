import numpy as np
import pandas as pd
from math import floor, ceil
import os.path as op



def dir_round(value, digits=3, direction='down'):
    if direction == 'up':
        return ceil(value * 10**digits) / 10**digits
    elif direction == 'down':
        return floor(value * 10**digits) / 10**digits
    else:
        return round(value, digits)


def butter(df, ):

    unneccessary_float_columns = [  # Unneccesary float columns
        'ID', 
        'TRACK_ID', 
        'POSITION_T', 
        'FRAME'
        ]

    # Loads the data into a DataFrame
    df = pd.DataFrame(df)

    # Reset the df index
    df = df.reset_index(drop=True)

    # Converts non-numeric values in selected columns to numeric values
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['POSITION_X', 'POSITION_Y', 'POSITION_T'])

    # Sorts the data in the DataFrame by TRACK_ID and POSITION_T (time position)
    df = df.sort_values(by=['TRACK_ID', 'POSITION_T'])    

    # For some reason, the y coordinates extracted from trackmate are mirrored. That ofcourse would not affect the statistical tests, only the data visualization. However, to not get mindfucked..
    # Reflect y-coordinates around the midpoint for the directionality to be accurate, according to the microscope videos.
    y_mid = (df['POSITION_Y'].min() + df['POSITION_Y'].max()) / 2
    df['POSITION_Y'] = 2 * y_mid - df['POSITION_Y']

    # The dataset itself has a very chaotic, multirow column "title system". Therefore in this list are again defined columns, which from now on will be used for consistency.
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

    # Droping all non numeric values, also dropping whole columns only containing non-numeric values.
    df = df.dropna(axis=1)

    # Here we convert the unnecessary floats (from the list in which we defined them) to integers
    df[unneccessary_float_columns] = df[unneccessary_float_columns].astype(int)

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

def calculate_track_lengths_and_net_distances(df):

    # Convert 'Track ID' to numeric (if it's not already)
    df['TRACK_ID'] = pd.to_numeric(df['TRACK_ID'], errors='coerce')

    # Making sure that no empty lines are created in the DataFrame
    if df.empty:
        return np.nan

    # Sum distances per Track ID
    track_lengths = df.groupby('TRACK_ID')['DISTANCE'].sum().reset_index()

    # Rename columns for clarity
    track_lengths.columns = ['TRACK_ID', 'TRACK_LENGTH']

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

    track_lengths_and_net_distances = pd.merge(track_lengths, net_distances, on='TRACK_ID', how='outer')

    # Return the results
    return pd.DataFrame(track_lengths_and_net_distances)

def calculate_confinement_ratio_for_each_cell(df):
    # Calculate the confinement ratio
    df['CONFINEMENT_RATIO'] = df['NET_DISTANCE'] / df['TRACK_LENGTH']
    Track_stats_df = df[['TRACK_ID','CONFINEMENT_RATIO']]
    return pd.DataFrame(Track_stats_df)

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

def weighted_mean_direction(angles, weights):
    weighted_sin = np.average(np.sin(angles), weights=weights)
    weighted_cos = np.average(np.cos(angles), weights=weights)
    return np.arctan2(weighted_sin, weighted_cos)
    
def weighted_std_deviation(angles, weights, mean_direction_rad):
    weighted_sin = np.average(np.sin(angles - mean_direction_rad), weights=weights)
    weighted_cos = np.average(np.cos(angles - mean_direction_rad), weights=weights)
    return np.sqrt(weighted_sin**2 + weighted_cos**2)
    
def weighted_median_direction(angles, weights):
    sorted_angles = np.sort(angles)
    sorted_weights = np.array(weights)[np.argsort(angles)]
    cumsum_weights = np.cumsum(sorted_weights)
    midpoint = np.sum(weights) / 2
    idx = np.searchsorted(cumsum_weights, midpoint)
    return sorted_angles[idx]

def calculate_absolute_directions_per_cell(df):
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

def calculate_weighted_directions_per_cell(df):
    df = df.dropna(subset=['TRACK_ID'])

    confinement_ratio_weighted_mean_direction_rad = df['MEAN_DIRECTION_RAD'] * df['CONFINEMENT_RATIO'] / df['CONFINEMENT_RATIO']
    confinement_ratio_weighted_mean_direction_deg = np.degrees(confinement_ratio_weighted_mean_direction_rad) % 360
    confinement_ratio_weighted_std_deviation_rad = df['STD_DEVIATION_RAD'] * df['CONFINEMENT_RATIO'] / df['CONFINEMENT_RATIO']    
    confinement_ratio_weighted_std_deviation_deg = np.degrees(confinement_ratio_weighted_std_deviation_rad) % 360
    confinement_ratio_weighted_median_direction_rad = df['MEADIAN_DIRECTION_RAD'] * df['CONFINEMENT_RATIO'] / df['CONFINEMENT_RATIO']
    confinement_ratio_weighted_median_direction_deg = np.degrees(confinement_ratio_weighted_median_direction_rad) % 360

    net_distance_weighted_mean_direction_rad = df['MEAN_DIRECTION_RAD'] * df['NET_DISTANCE'] / df['NET_DISTANCE']
    net_distance_weighted_mean_direction_deg = np.degrees(net_distance_weighted_mean_direction_rad) % 360
    net_distance_weighted_std_deviation_rad = df['STD_DEVIATION_RAD'] * df['NET_DISTANCE'] / df['NET_DISTANCE']
    net_distance_weighted_std_deviation_deg = np.degrees(net_distance_weighted_std_deviation_rad) % 360
    net_distance_weighted_median_direction_rad = df['MEADIAN_DIRECTION_RAD'] * df['NET_DISTANCE'] / df['NET_DISTANCE']
    net_distance_weighted_median_direction_deg = np.degrees(net_distance_weighted_median_direction_rad) % 360    

    return pd.DataFrame({
    'TRACK_ID': df.index, 
    'MEAN_DIRECTION_DEG_weight_confinement': confinement_ratio_weighted_mean_direction_deg, 
    'STD_DEVIATION_DEG_weight_confinement': confinement_ratio_weighted_std_deviation_deg, 
    'MEDIAN_DIRECTION_DEG_weight_confinement': confinement_ratio_weighted_median_direction_deg, 
    'MEAN_DIRECTION_RAD_weight_confinement': confinement_ratio_weighted_mean_direction_rad, 
    'STD_DEVIATION_RAD_weight_confinement': confinement_ratio_weighted_std_deviation_rad, 
    'MEADIAN_DIRECTION_RAD_weight_confinement': confinement_ratio_weighted_median_direction_rad,
    'MEAN_DIRECTION_DEG_weight_net_dis': net_distance_weighted_mean_direction_deg, 
    'STD_DEVIATION_DEG_weight_net_dis': net_distance_weighted_std_deviation_deg, 
    'MEDIAN_DIRECTION_DEG_weight_net_dis': net_distance_weighted_median_direction_deg, 
    'MEAN_DIRECTION_RAD_weight_net_dis': net_distance_weighted_mean_direction_rad, 
    'STD_DEVIATION_RAD_weight_net_dis': net_distance_weighted_std_deviation_rad, 
    'MEADIAN_DIRECTION_RAD_weight_net_dis': net_distance_weighted_median_direction_rad
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
        'MEAN_DIRECTION_DEG': mean_direction_deg, 
        'STD_DEVIATION_DEG': std_deviatin_deg, 
        'MEDIAN_DIRECTION_DEG': median_direction_deg, 
        'MEAN_DIRECTION_RAD': mean_direction_rad, 
        'STD_DEVIATION_RAD': std_deviation_rad, 
        'MEADIAN_DIRECTION_RAD': median_direction_rad
        }).reset_index(drop=True)

def calculate_weighted_directions_per_frame(df):
    grouped = df.groupby('POSITION_T')
    
    # Compute weighted metrics
    weighted_mean_direction_rad = grouped.apply(lambda x: weighted_mean_direction(x['DIRECTION_RAD'], x['DISTANCE'])).reset_index(level=0, drop=True)
    weighted_mean_direction_deg = np.degrees(weighted_mean_direction_rad) % 360
    
    weighted_std_deviation_rad = grouped.apply(lambda x: weighted_std_deviation(x['DIRECTION_RAD'], x['DISTANCE'], weighted_mean_direction(x['DIRECTION_RAD'], x['DISTANCE']))).reset_index(level=0, drop=True)
    weighted_std_deviation_deg = np.degrees(weighted_std_deviation_rad) % 360
    
    weighted_median_direction_rad = grouped.apply(lambda x: weighted_median_direction(x['DIRECTION_RAD'], x['DISTANCE'])).reset_index(level=0, drop=True)
    weighted_median_direction_deg = np.degrees(weighted_median_direction_rad) % 360
    
    result_df = pd.DataFrame({
        'POSITION_T': weighted_mean_direction_rad.index,
        'MEAN_DIRECTION_DEG_weight_mean_dis': weighted_mean_direction_deg,
        'STD_DEVIATION_DEG_weight_mean_dis': weighted_std_deviation_deg,
        'MEDIAN_DIRECTION_DEG_weight_mean_dis': weighted_median_direction_deg,
        'MEAN_DIRECTION_RAD_weight_mean_dis': weighted_mean_direction_rad,
        'STD_DEVIATION_RAD_weight_mean_dis': weighted_std_deviation_rad,
        'MEDIAN_DIRECTION_RAD_weight_mean_dis': weighted_median_direction_rad
    }).reset_index(drop=True)

    result_df['POSITION_T'] = result_df['POSITION_T'] + 1
    
    return result_df

def calculate_number_of_frames_per_cell(spot_stats_df):
    # Count the number of frames for each TRACK_ID in the Spot_stats2_df
    frames_per_track = (spot_stats_df.groupby("TRACK_ID").size().reset_index(name="NUM_FRAMES"))

    return frames_per_track

def calculate_speed(df, variable):

    min_speed_microns_min = df.groupby(variable)['DISTANCE'].min().reset_index()
    min_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MIN'}, inplace=True)
    max_speed_microns_min = df.groupby(variable)['DISTANCE'].max().reset_index()
    max_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MAX'}, inplace=True)
    mean_speed_microns_min = df.groupby(variable)['DISTANCE'].mean().reset_index()
    mean_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MEAN'}, inplace=True)
    std_deviation_speed_microns_min = df.groupby(variable)['DISTANCE'].std().reset_index()
    std_deviation_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_STD_DEVIATION'}, inplace=True)
    median_speed_microns_min = df.groupby(variable)['DISTANCE'].median().reset_index()
    median_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MEDIAN'}, inplace=True)
    merge = pd.merge(min_speed_microns_min, max_speed_microns_min, on=variable)
    merge = pd.merge(merge, mean_speed_microns_min, on=variable)
    merge = pd.merge(merge, std_deviation_speed_microns_min, on=variable)
    merged = pd.merge(merge, median_speed_microns_min, on=variable)

    return pd.DataFrame(merged)

def merge_dfs(dataframes, on):

    # Initialize the first DataFrame as the base for merging
    merged_df = dataframes[0]

    # Use a for loop to merge each subsequent DataFrame
    for df in dataframes[1:]:
        # Merge, avoiding duplicate columns
        merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
        merged_df = pd.merge(
            merged_df,
            df[merge_columns],  # Select only necessary columns from df
            on=on,
            how='outer'
        )

    return merged_df


def percentile_thresholding(df, column_name: str, values: tuple):
    try:
        lower_percentile, upper_percentile = values
        lower_threshold = np.percentile(df[column_name], lower_percentile)
        upper_threshold = np.percentile(df[column_name], upper_percentile)
        return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]
    except ValueError:
        return df

def literal_thresholding(df, column_name: str, values: tuple):
    lower_threshold, upper_threshold = values
    return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]

def dataframe_filter(df, df_filter):
    return df[df["TRACK_ID"].isin(df_filter["TRACK_ID"])]

def values_for_a_metric(df, metric):
    df.dropna()
    min_value = floor(df[metric].min())
    max_value = ceil(df[metric].max())
    return min_value, max_value



def filename_get(path):
    return op.basename(path)