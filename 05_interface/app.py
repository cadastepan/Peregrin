from shiny import reactive
from shiny.express import input, render, ui
from shiny.types import FileInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.signal import savgol_filter
from peregrin.scripts import PlotParams
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde


# ===========================================================================================================================================================================================================================================================================
# Page specs and layout
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ui.page_opts(
    title="Peregrin", 
    fillable=False
    )



# ===========================================================================================================================================================================================================================================================================
# Creating reactive variables for dataframe storage
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

buttered_df = reactive.value()
Spot_stats_df = reactive.value()
Track_stats_df = reactive.value()
Frame_stats_df = reactive.value()



# ===========================================================================================================================================================================================================================================================================
# Data panel
# Reading a selected CSV file and cleaning it
# Extracting separate dataframes
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with ui.nav_panel("Data"):  # Data panel
    
    ui.input_file("file1", "Input CSV", accept=[".csv"], multiple=False)   # Input field

    @reactive.calc # Decorator for a reactive function
    def parsed_file(): # File reading
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        df = pd.read_csv(file[0]["datapath"])  # pyright: ignore[reportUnknownMemberType]


        # ===========================================================================================================================================================================================================================================================================
        # File cleaning
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        unneccessary_float_columns = [  # Unneccesary float columns
            
            'ID', 
            'TRACK_ID', 
            'POSITION_T', 
            'FRAME'
            ]
        
        def butter(df, float_columns): # File cleaning

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
            df[float_columns] = df[float_columns].astype(int)

            return df
        buttered = butter(df, unneccessary_float_columns)

        return buttered



    # =============================================================================================================================================================================================================================================================================
    # Functions handling buttered data and extracting separate dataframes with Spot, Track and Frame statistics
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
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
        # Weighted median approximation: using percentile function from scipy
        from scipy.stats import cumfreq
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



    # =============================================================================================================================================================================================================================================================================
    # Executing the functions 
    # Creating separate dataframes
    # Itermidiate caching of the dataframes
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @reactive.effect
    def update_buttered_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        df = parsed_file()  # Call the expensive function.
        buttered_df.set(df)


    @reactive.calc
    def process_spot_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        buttered = buttered_df.get()

        distances_for_each_cell_per_frame_df = calculate_traveled_distances_for_each_cell_per_frame(buttered) # Call the function to calculate distances for each cell per frame and create the Spot_statistics .csv file
        direction_for_each_cell_per_frame_df = calculate_direction_of_travel_for_each_cell_per_frame(buttered) # Call the function to calculate direction_for_each_cell_per_frame_df

        Spot_stats_dfs = [buttered, distances_for_each_cell_per_frame_df, direction_for_each_cell_per_frame_df]

        Spot_stats = merge_dfs(Spot_stats_dfs, on=['TRACK_ID', 'POSITION_T'])
        # Spot_stats.to_csv('Spot_stats.csv') # Saving the Spot_stats DataFrame into a newly created .csv file

        return Spot_stats

    @reactive.effect
    def update_Spot_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        Spot_stats = process_spot_data()
        Spot_stats_df.set(Spot_stats)


    @reactive.calc
    def process_track_data2():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()

        Spot_stats = Spot_stats_df.get()

        tracks_lengths_and_net_distances_df = calculate_track_lengths_and_net_distances(Spot_stats) # Calling function to calculate the total distance traveled for each cell from the distances_for_each_cell_per_frame_df
        confinement_ratios_df = calculate_confinement_ratio_for_each_cell(tracks_lengths_and_net_distances_df) # Call the function to calculate confinement ratios from the Track_statistics1_df and write it into the Track_statistics1_df
        track_directions_df = calculate_absolute_directions_per_cell(Spot_stats) # Call the function to calculate directions_per_cell_df
        frames_per_track = calculate_number_of_frames_per_cell(Spot_stats)
        speeds_per_cell = calculate_speed(Spot_stats, 'TRACK_ID')

        Track_stats_dfs = [tracks_lengths_and_net_distances_df, confinement_ratios_df, track_directions_df, frames_per_track, speeds_per_cell]
        Track_stats = merge_dfs(Track_stats_dfs, on='TRACK_ID')
        # Track_stats.to_csv('Track_stats.csv', index=False) # Save the Track_stats created DataFrame into a newly created Track_stats_debugging.csv file

        return Track_stats

    
    @reactive.calc
    def process_frame_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()

        Spot_stats = Spot_stats_df.get()
        
        distances_per_frame_df = calculate_distances_per_frame(Spot_stats) # Call the function to calculate distances_per_frame_df
        absolute_directions_per_frame_df = calculate_absolute_directions_per_frame(Spot_stats) # Call the function to calculate directions_per_frame_df
        weighted_directions_per_frame = calculate_weighted_directions_per_frame(Spot_stats) # Call the function tp calculate weighted_directions_per_frame
        speeds_per_frame = calculate_speed(Spot_stats, 'POSITION_T')

        Frame_stats_dfs = [distances_per_frame_df, absolute_directions_per_frame_df, weighted_directions_per_frame, speeds_per_frame]

        Frame_stats = merge_dfs(Frame_stats_dfs, on='POSITION_T')
        Frame_stats = Frame_stats.fillna(0)
        # Frame_stats.to_csv('Frame_stats.csv', index=False) # Save the Frame_stats created DataFrame into a newly created Frame_stats_debugging.csv file

        return Frame_stats
    

    @reactive.effect
    def update_Track_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        Track_stats = process_track_data2()
        Track_stats_df.set(Track_stats) 

    @reactive.effect
    def update_Frame_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        Frame_stats = process_frame_data()
        Frame_stats_df.set(Frame_stats)


    
    # =============================================================================================================================================================================================================================================================================
    # Separately displaying the dataframes
    # Enabling the user to download the dataframes as .csv files
    # Enabling data filtering?
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    with ui.layout_columns():  
        with ui.card():  
            ui.card_header("Spot stats")

            @render.data_frame
            def render_spot_stats():
                file: list[FileInfo] | None = input.file1()
                if file is None:
                    return pd.DataFrame()
                
                Spot_stats = Spot_stats_df.get()
                return render.DataGrid(Spot_stats)
            
        
        with ui.card():
            ui.card_header("Track stats")
            
            @render.data_frame
            def render_track_stats():
                file: list[FileInfo] | None = input.file1()
                if file is None:
                    return pd.DataFrame()
                
                Track_stats = Track_stats_df.get()
                return render.DataGrid(Track_stats)
            
            
        with ui.card():
            ui.card_header("Frame stats")

            @render.data_frame
            def render_frame_stats():
                file: list[FileInfo] | None = input.file1()
                if file is None:
                    return pd.DataFrame()
                
                Frame_stats = Frame_stats_df.get()
                return render.DataGrid(Frame_stats)
    



# ===========================================================================================================================================================================================================================================================================
# Tracks panel
# Track visualization
# Plotting track statistics
# Statistical testing
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with ui.nav_panel("Tracks"):


    # ===========================================================================================================================================================================================================================================================================
    # Optics parameters
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Definition of micron length per pixel
    microns_per_pixel = 0.7381885238402274 # for 10x lens

    # Define the desired dimensions in microns
    x_min, x_max = 0, (1600 * microns_per_pixel)
    y_min, y_max = 0, (1200 * microns_per_pixel)
    x_axe_remainder = x_max-1150
    x_add = 50 - x_axe_remainder
    y_ax_remainder = y_max-850
    x_substract = (x_max - y_max) + (y_ax_remainder - 50)

    # Calculate the aspect ratio
    aspect_ratio = x_max / y_max


    # ===========================================================================================================================================================================================================================================================================
    # Plot specs
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    title_size = 18
    label_size = 11
    figtext_size = 9
    compass_annotations_size = 15
    figtext_color = 'grey'


    # ===========================================================================================================================================================================================================================================================================
    # Plotting functions
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        ax.set_facecolor('dimgrey')
        ax.grid(True, which='both', axis='both', color='grey', linewidth=1)

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

    def visualize_full_tracks(df, df2, threshold, lw=1):  #Trakcs visualisation

        # Recognizing the presence of a threshold
        if threshold == None:
            threshold = '_no_threshold'
        else:
            threshold = '_' + threshold

        # Using the  track_visuals function
        fig_visuals, ax_visuals, track_ids_visuals, track_colors_visuals, norm_visuals, colormap_visuals = track_visuals(df, df2)

        # Plot the full tracks
        for track_id in track_ids_visuals:
            track_data = df[df['TRACK_ID'] == track_id]
            x = track_data['POSITION_X']
            y = track_data['POSITION_Y']
            ax_visuals.plot(x, y, lw=lw, color=track_colors_visuals[track_id], label=f'Track {track_id}')
            
            # Get the original color from track_colors_visuals[track_id]
            original_color = mcolors.to_rgb(track_colors_visuals[track_id])
            # Darken the color by reducing the brightness (by scaling each RGB channel)
            darkened_color = np.array(original_color) * 0.7  # Adjust 0.7 to a value between 0 and 1 for different darkness levels
            # Ensure that no channel goes below 0 (clip values if necessary)
            darkened_color = np.clip(darkened_color, 0, 1)
            # Apply the darkened color with a slight increase in the green and blue channels for the original factor you had (optional)
            color = darkened_color * np.array([1.0, 1.0, 0.8])  # If you want to keep the original adjustment
        

            if len(x) > 1:
                # Add arrow to indicate direction
                dx = x.diff().iloc[-1]
                dy = y.diff().iloc[-1]
                if dx != 0 or dy != 0:
                    # Create an arrow instead of a circle
                    arrow = FancyArrowPatch(
                        posA=(x.iloc[-2], y.iloc[-2]),  # Start position (second-to-last point)
                        posB=(x.iloc[-1], y.iloc[-1]),  # End position (last point)
                        arrowstyle='-|>',  # Style of the arrow (you can adjust the style as needed)
                        color=color,  # Set the color of the arrow
                        mutation_scale=5,  # Scale the size of the arrow head (adjust this based on the plot scale)
                        linewidth=1.2,  # Line width for the arrow
                        zorder=10  # Ensure the arrow is drawn on top of the line
                    )

                    # Add the arrow to your plot (if you're using a `matplotlib` figure/axes)
                    plt.gca().add_patch(arrow)

        # plt.savefig(f'01a_Full_tracks_snapshot{threshold}.png', dpi=300)
        # plt.figure()
        return ax_visuals

    def visualize_smoothened_tracks(df, df2, threshold, smoothing_type=None, smoothing_index=10, lw=1):  # smoothened tracks visualization

        # Recognizing the presence of a threshold
        if threshold is None:
            threshold = '_no_threshold'
        else:
            threshold = '_' + threshold

        # Using the track_visuals function
        fig_visuals, ax_visuals, track_ids_visuals, track_colors_visuals, norm_visuals, colormap_visuals = track_visuals(df, df2)

        # Plot the full tracks
        for track_id in track_ids_visuals:
            track_data = df[df['TRACK_ID'] == track_id]
            x = track_data['POSITION_X']
            y = track_data['POSITION_Y']
            
            # Apply smoothing to the track (if applicable)
            if smoothing_type == 'moving_average':
                x_smoothed = x.rolling(window=smoothing_index, min_periods=1).mean()
                y_smoothed = y.rolling(window=smoothing_index, min_periods=1).mean()
            else:
                x_smoothed = x
                y_smoothed = y

            ax_visuals.plot(x_smoothed, y_smoothed, lw=lw, color=track_colors_visuals[track_id], label=f'Track {track_id}')

            # Get the original color from track_colors_visuals[track_id]
            original_color = mcolors.to_rgb(track_colors_visuals[track_id])
            # Darken the color by reducing the brightness (by scaling each RGB channel)
            darkened_color = np.array(original_color) * 0.7  # Adjust 0.7 to a value between 0 and 1 for different darkness levels
            # Ensure that no channel goes below 0 (clip values if necessary)
            darkened_color = np.clip(darkened_color, 0, 1)
            # Apply the darkened color with a slight increase in the green and blue channels for the original factor you had (optional)
            color = darkened_color * np.array([1.0, 1.0, 0.8])  # If you want to keep the original adjustment


            if len(x_smoothed) > 1:
                # Extract the mean direction from df2 for the current track
                mean_direction_rad = df2[df2['TRACK_ID'] == track_id]['MEAN_DIRECTION_RAD'].values[0]
                
                # Use trigonometry to calculate the direction (dx, dy) from the angle
                dx = np.cos(mean_direction_rad)  # Change in x based on angle
                dy = np.sin(mean_direction_rad)  # Change in y based on angle
                
                # Create an arrow to indicate direction
                arrow = FancyArrowPatch(
                    posA=(x_smoothed.iloc[-2], y_smoothed.iloc[-2]),  # Start position (second-to-last point)
                    posB=(x_smoothed.iloc[-2] + dx, y_smoothed.iloc[-2] + dy),  # End position based on direction
                    arrowstyle='-|>',  # Style of the arrow (you can adjust the style as needed)
                    color=color,  # Set the color of the arrow
                    mutation_scale=5,  # Scale the size of the arrow head (adjust this based on the plot scale)
                    linewidth=1.2,  # Line width for the arrow
                    zorder=30  # Ensure the arrow is drawn on top of the line
                )

                # Add the arrow to your plot (if you're using a `matplotlib` figure/axes)
                plt.gca().add_patch(arrow)

        # plt.savefig(f'01a_Full_tracks_snapshot{threshold}.png', dpi=300)
        # plt.show()
        return ax_visuals






    with ui.navset_card_tab():
        with ui.nav_panel("Visualization"):
            with ui.card():
                ui.card_header("Full tracks visualization")
                @render.plot
                def plot1():
                    return visualize_full_tracks(df=Spot_stats_df.get(), df2=Track_stats_df.get(), threshold=None, lw=0.5)

            with ui.card():
                ui.card_header("Smoothened tracks visualization")
                @render.plot
                def plot2():
                    return visualize_smoothened_tracks(df=Spot_stats_df.get(), df2=Track_stats_df.get(), threshold=None, smoothing_type='moving_average', smoothing_index=50, lw=0.8)










with ui.nav_panel("Frames"):
    with ui.navset_card_tab():
        with ui.nav_panel("Visualization"):
            "content"
        with ui.nav_panel("Tests"):
            "content"


