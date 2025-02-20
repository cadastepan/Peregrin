from shiny import reactive
from shiny.express import input, render, ui
from shiny.types import FileInfo

import pandas as pd
import numpy as np
import io

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from peregrin.scripts import PlotParams
import utils.data_utils as du
import utils.plot_utils as pu


# ===========================================================================================================================================================================================================================================================================
# Page specs and layout

ui.page_opts(
    title="Peregrin", 
    fillable=False
    )


# ===========================================================================================================================================================================================================================================================================
# Creating reactive variables for raw dataframe storage

raw_Buttered_df = reactive.value()
raw_Spot_stats_df = reactive.value()
raw_Track_stats_df = reactive.value()
raw_Frame_stats_df = reactive.value()


# ===========================================================================================================================================================================================================================================================================
# Creating reactive variables for processed dataframe storage

Buttered_df = reactive.value()
Spot_stats_df = reactive.value()
Track_stats_df = reactive.value()
Frame_stats_df = reactive.value()


# ===========================================================================================================================================================================================================================================================================
# Creating other reactive variables 

slider_values = reactive.value()   # Creating a rective value for the slider values



# ===========================================================================================================================================================================================================================================================================
# Data panel
# ===========================================================================================================================================================================================================================================================================
# Reading a selected CSV file and cleaning it
# Extracting separate dataframes


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
        buttered = du.butter(df, unneccessary_float_columns) # converting unnecessary float columns to int

        return buttered


    # =============================================================================================================================================================================================================================================================================
    # Executing the functions 
    # Creating separate dataframes
    # Itermidiate caching of the dataframes

    @reactive.effect
    def update_buttered_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        df = parsed_file()  # Call the expensive function.
        raw_Buttered_df.set(df)

    @reactive.calc
    def process_spot_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        buttered = raw_Buttered_df.get()

        distances_for_each_cell_per_frame_df = du.calculate_traveled_distances_for_each_cell_per_frame(buttered) # Call the function to calculate distances for each cell per frame and create the Spot_statistics .csv file
        direction_for_each_cell_per_frame_df = du.calculate_direction_of_travel_for_each_cell_per_frame(buttered) # Call the function to calculate direction_for_each_cell_per_frame_df

        Spot_stats_dfs = [buttered, distances_for_each_cell_per_frame_df, direction_for_each_cell_per_frame_df]

        Spot_stats = du.merge_dfs(Spot_stats_dfs, on=['TRACK_ID', 'POSITION_T'])
        # Spot_stats.to_csv('Spot_stats.csv') # Saving the Spot_stats DataFrame into a newly created .csv file

        return Spot_stats

    @reactive.effect
    def update_Spot_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        Spot_stats = process_spot_data()
        raw_Spot_stats_df.set(Spot_stats)
        Spot_stats_df.set(Spot_stats)


    @reactive.calc
    def process_track_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()

        Spot_stats = raw_Spot_stats_df.get()

        tracks_lengths_and_net_distances_df = du.calculate_track_lengths_and_net_distances(Spot_stats) # Calling function to calculate the total distance traveled for each cell from the distances_for_each_cell_per_frame_df
        confinement_ratios_df = du.calculate_confinement_ratio_for_each_cell(tracks_lengths_and_net_distances_df) # Call the function to calculate confinement ratios from the Track_statistics1_df and write it into the Track_statistics1_df
        track_directions_df = du.calculate_absolute_directions_per_cell(Spot_stats) # Call the function to calculate directions_per_cell_df
        frames_per_track = du.calculate_number_of_frames_per_cell(Spot_stats)
        speeds_per_cell = du.calculate_speed(Spot_stats, 'TRACK_ID')

        Track_stats_dfs = [tracks_lengths_and_net_distances_df, confinement_ratios_df, track_directions_df, frames_per_track, speeds_per_cell]
        Track_stats = du.merge_dfs(Track_stats_dfs, on='TRACK_ID')
        Track_stats['CONFINEMENT_RATIO'] = Track_stats['NET_DISTANCE'] / Track_stats['TRACK_LENGTH']
        Track_stats = du.merge_dfs(Track_stats_dfs, on='TRACK_ID')
        # Track_stats.to_csv('Track_stats.csv', index=False) # Save the Track_stats created DataFrame into a newly created Track_stats_debugging.csv file

        return Track_stats

    
    @reactive.calc
    def process_frame_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()

        Spot_stats = Spot_stats_df.get()
        
        distances_per_frame_df = du.calculate_distances_per_frame(Spot_stats) # Call the function to calculate distances_per_frame_df
        absolute_directions_per_frame_df = du.calculate_absolute_directions_per_frame(Spot_stats) # Call the function to calculate directions_per_frame_df
        weighted_directions_per_frame = du.calculate_weighted_directions_per_frame(Spot_stats) # Call the function tp calculate weighted_directions_per_frame
        speeds_per_frame = du.calculate_speed(Spot_stats, 'POSITION_T')

        Frame_stats_dfs = [distances_per_frame_df, absolute_directions_per_frame_df, weighted_directions_per_frame, speeds_per_frame]

        Frame_stats = du.merge_dfs(Frame_stats_dfs, on='POSITION_T')
        Frame_stats = Frame_stats.fillna(0)
        # Frame_stats.to_csv('Frame_stats.csv', index=False) # Save the Frame_stats created DataFrame into a newly created Frame_stats_debugging.csv file

        return Frame_stats
    

    @reactive.effect
    def update_Track_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        Track_stats = process_track_data()
        raw_Track_stats_df.set(Track_stats)
        Track_stats_df.set(Track_stats)

    @reactive.effect
    def update_Frame_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        Frame_stats = process_frame_data()
        raw_Frame_stats_df.set(Frame_stats)
        Frame_stats_df.set(Frame_stats)


    # =============================================================================================================================================================================================================================================================================
    # Separately displaying the dataframes
    # Enabling the user to download the dataframes as .csv files
    # Enabling data filtering?

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
# Sidebar
# ===========================================================================================================================================================================================================================================================================



with ui.sidebar(open="open", position="right", bg="f8f8f8"): 


    # ===========================================================================================================================================================================================================================================================================
    # Creating a possibility for thresholding metric selection
    # Creating a possibility for thresholding filter selection
    # Creating a slider for thresholding

    ui.input_select(  
        "metric",  
        "Thresholding metric:",  
        {
            "TRACK_LENGTH": "Track length", 
            "NET_DISTANCE": "Net distance", 
            "CONFINEMENT_RATIO": "Confinement ratio",
            "NUM_FRAMES": "Number of frames",
            "SPEED_MEAN": "Mean speed",
            "SPEED_MEDIAN": "Median speed",
            "SPEED_MAX": "Max speed",
            "SPEED_MIN": "Min speed",
            "SPEED_STD_DEVIATION": "Speed standard deviation",
            "MEAN_DIRECTION_DEG": "Mean direction (degrees)",
            "MEAN_DIRECTION_RAD": "Mean direction (radians)",
            "STD_DEVIATION_DEG": "Standard deviation (degrees)",
            "STD_DEVIATION_RAD": "Standard deviation (radians)",

            "POSITION_T": "Position t",
            "POSITION_X": "Position x",
            "POSITION_Y": "Position y",
            "QUALITY": "Quality",
            "VISIBILITY": "Visibility"

            },  
    )  

    ui.input_select(
        "filter",
        "Thresholding filter:",
        {
            "literal": "Literal",
            "percentile": "Percentile",
            },
    )

    ui.input_slider(
        "slider",
        "Threshold",
        min=0,
        max=100,
        value=(0, 100)
    )


    # ===========================================================================================================================================================================================================================================================================
    # Reactive functions updating the slider values
    
    @reactive.effect
    def update_slider_values():
        if input.metric() in ["TRACK_LENGTH", "NET_DISTANCE", "CONFINEMENT_RATIO", "NUM_FRAMES", "SPEED_MEAN", "SPEED_MEDIAN", "SPEED_MAX", "SPEED_MIN", "SPEED_STD_DEVIATION", "MEAN_DIRECTION_DEG", "MEAN_DIRECTION_RAD", "STD_DEVIATION_DEG", "STD_DEVIATION_RAD"]:
            if raw_Track_stats_df.get().empty:
                slider_values.set(0, 100)
            else:
                values = du.values_for_a_metric(raw_Track_stats_df.get(), input.metric())
                slider_values.set(values)
        elif input.metric() in ["POSITION_T", "POSITION_X", "POSITION_Y", "QUALITY", "VISIBILITY"]:
            if raw_Spot_stats_df.get().empty:
                slider_values.set(0, 100)
            else:
                values = du.values_for_a_metric(raw_Spot_stats_df.get(), input.metric())
                slider_values.set(values)
        else:
            return slider_values.set(0, 100)

    @reactive.effect
    def update_slider():
        if input.filter() == "percentile":
            ui.update_slider("slider", min=0, max=100, value=(0, 100), step=1)
        elif input.filter() == "literal":
            values = slider_values.get()
            range = values[1] - values[0]

            if range <= 10:
                steps = 0.01
            elif range <= 100:
                steps = 0.1
            else:
                steps = 1
            
            if values:
                ui.update_slider("slider", min=values[0], max=values[1], value=values, step=steps)

    # @render.plot
    # def threshold_histogram():
    #     metric = input.metric()
    #     filter_type = input.filter()
    #     slider_range = input.slider()

    #     if metric in ["TRACK_LENGTH", "NET_DISTANCE", "CONFINEMENT_RATIO", "NUM_FRAMES", "SPEED_MEAN", "SPEED_MEDIAN", "SPEED_MAX", "SPEED_MIN", "SPEED_STD_DEVIATION", "MEAN_DIRECTION_DEG", "MEAN_DIRECTION_RAD", "STD_DEVIATION_DEG", "STD_DEVIATION_RAD"]:
    #         data = raw_Track_stats_df.get()
    #     elif metric in ["POSITION_T", "POSITION_X", "POSITION_Y", "QUALITY", "VISIBILITY"]:
    #         data = raw_Spot_stats_df.get()
    #     else:
    #         return plt.figure()

    #     if data.empty:
    #         return plt.figure()

    #     values = data[metric].dropna()

    #     if filter_type == "percentile":
    #         lower_percentile = np.percentile(values, slider_range[0])
    #         upper_percentile = np.percentile(values, slider_range[1])
    #         filtered_values = values[(values >= lower_percentile) & (values <= upper_percentile)]
    #     else:
    #         filtered_values = values[(values >= slider_range[0]) & (values <= slider_range[1])]

    #     fig, ax = plt.subplots()
    #     ax.hist(filtered_values, bins=25)
    #     ax.set_xticks([])  # Remove x-axis ticks
    #     ax.set_yticks([])  # Remove y-axis ticks

    #     return fig


    @render.plot
    def threshold_histogram():
        metric = input.metric()
        filter_type = input.filter()
        slider_range = input.slider()

        if metric in ["TRACK_LENGTH", "NET_DISTANCE", "CONFINEMENT_RATIO", "NUM_FRAMES", "SPEED_MEAN", "SPEED_MEDIAN", "SPEED_MAX", "SPEED_MIN", "SPEED_STD_DEVIATION", "MEAN_DIRECTION_DEG", "MEAN_DIRECTION_RAD", "STD_DEVIATION_DEG", "STD_DEVIATION_RAD"]:
            data = raw_Track_stats_df.get()
        elif metric in ["POSITION_T", "POSITION_X", "POSITION_Y", "QUALITY", "VISIBILITY"]:
            data = raw_Spot_stats_df.get()
        else:
            return plt.figure()

        if data.empty:
            return plt.figure()

        values = data[metric].dropna()

        if filter_type == "percentile":
            lower_percentile = np.percentile(values, slider_range[0])
            upper_percentile = np.percentile(values, slider_range[1])
            lower_bound = lower_percentile
            upper_bound = upper_percentile
        else:
            lower_bound = slider_range[0]
            upper_bound = slider_range[1]

        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(values, bins=40)

        for i in range(len(patches)):
            if bins[i] < lower_bound or bins[i+1] > upper_bound:
                patches[i].set_facecolor('grey')
            else:
                patches[i].set_facecolor('#337ab7')

        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

        return fig


    # ===========================================================================================================================================================================================================================================================================
    # Thresholding the data based on percentiles
    
    @reactive.calc
    def thresholded_data():
        if input.filter() == "percentile":
            if input.metric() in ["TRACK_LENGTH", "NET_DISTANCE", "CONFINEMENT_RATIO", "NUM_FRAMES", "SPEED_MEAN", "SPEED_MEDIAN", "SPEED_MAX", "SPEED_MIN", "SPEED_STD_DEVIATION", "MEAN_DIRECTION_DEG", "MEAN_DIRECTION_RAD", "STD_DEVIATION_DEG", "STD_DEVIATION_RAD"]:
                return du.percentile_thresholding(raw_Track_stats_df.get(), input.metric(), input.slider())
            elif input.metric() in ["POSITION_T", "POSITION_X", "POSITION_Y", "QUALITY", "VISIBILITY"]:
                return du.percentile_thresholding(raw_Spot_stats_df.get(), input.metric(), input.slider())
        elif input.filter() == "literal":
            if input.metric() in ["TRACK_LENGTH", "NET_DISTANCE", "CONFINEMENT_RATIO", "NUM_FRAMES", "SPEED_MEAN", "SPEED_MEDIAN", "SPEED_MAX", "SPEED_MIN", "SPEED_STD_DEVIATION", "MEAN_DIRECTION_DEG", "MEAN_DIRECTION_RAD", "STD_DEVIATION_DEG", "STD_DEVIATION_RAD"]:
                return du.literal_thresholding(raw_Track_stats_df.get(), input.metric(), input.slider())
            elif input.metric() in ["POSITION_T", "POSITION_X", "POSITION_Y", "QUALITY", "VISIBILITY"]:
                return du.literal_thresholding(raw_Spot_stats_df.get(), input.metric(), input.slider())

    @reactive.effect
    def update_thresholded_data():
        thresholded_df = thresholded_data()
        if input.metric() in ["TRACK_LENGTH", "NET_DISTANCE", "CONFINEMENT_RATIO", "NUM_FRAMES", "SPEED_MEAN", "SPEED_MEDIAN", "SPEED_MAX", "SPEED_MIN", "SPEED_STD_DEVIATION", "MEAN_DIRECTION_DEG", "MEAN_DIRECTION_RAD", "STD_DEVIATION_DEG", "STD_DEVIATION_RAD"]:
            Track_stats_df.set(thresholded_df)
            Spot_stats_df.set(du.dataframe_filter(raw_Spot_stats_df.get(), Track_stats_df.get()))
        elif input.metric() in ["POSITION_T", "POSITION_X", "POSITION_Y", "QUALITY", "VISIBILITY"]:
            Spot_stats_df.set(thresholded_df)
            Track_stats_df.set(du.dataframe_filter(raw_Track_stats_df.get(), Spot_stats_df.get()))





# ===========================================================================================================================================================================================================================================================================	
# Optics parameters

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
# Globally used callables

# plot specs
title_size = 16
title_size2 = 12
label_size = 11
figtext_size = 9
compass_annotations_size = 15
figtext_color = 'grey'

# Color maps
cmap_cells = mcolors.LinearSegmentedColormap.from_list("", ["#9b598910", "#9b181eff"])
cmap_frames = plt.get_cmap('viridis')




# ===========================================================================================================================================================================================================================================================================
# Visualisation panel
# ===========================================================================================================================================================================================================================================================================
# Data visualisation
# Plots
# Statistical testing?


with ui.nav_panel("Visualisation"):


    # ===========================================================================================================================================================================================================================================================================
    # Tracks tab

    with ui.navset_card_pill():
        with ui.nav_panel("Tracks"):

            with ui.navset_card_tab(id="tab1"):
                with ui.nav_panel("Track visualisation"):
                    with ui.layout_columns(
                        col_widths=(6,6,6,6),
                        row_heights=(3, 4),	
                    ):
        
                        with ui.card(full_screen=True):
                            ui.card_header("Raw tracks visualization")
                            @render.plot
                            def plot1():
                                return pu.visualize_full_tracks(
                                    df=Spot_stats_df.get(), 
                                    df2=Track_stats_df.get(), 
                                    threshold=None, 
                                    lw=0.5
                                    )

                        with ui.card(full_screen=True):
                            ui.card_header("Smoothened tracks visualization")
                            @render.plot
                            def plot2():
                                return pu.visualize_smoothened_tracks(
                                    df=Spot_stats_df.get(), 
                                    df2=Track_stats_df.get(), 
                                    threshold=None, 
                                    smoothing_type='moving_average', 
                                    smoothing_index=50, 
                                    lw=0.8
                                    )

                with ui.nav_panel("Directionality plots"):
                    with ui.layout_columns():
                        with ui.card(full_screen=True):  
                            ui.card_header("Directionality")
                            with ui.layout_column_wrap(width=1 / 2):
                                with ui.card(full_screen=False):
                                    ui.card_header("Scaled by confinement ratio")
                                    @render.plot
                                    def plot3():
                                        figure = pu.migration_directions_with_kde_plus_mean(
                                            df=Track_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Cells', 
                                            scaling_metric='CONFINEMENT_RATIO', 
                                            cmap_normalization_metric=None, 
                                            cmap=cmap_cells, 
                                            threshold=None,
                                            title_size2=title_size2
                                            )
                                        return figure
                                    
                                    @render.download(label="Download", filename="Track directionality.png")
                                    def download1():
                                        figure = pu.migration_directions_with_kde_plus_mean(
                                            df=Track_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Cells', 
                                            scaling_metric='CONFINEMENT_RATIO', 
                                            cmap_normalization_metric=None, 
                                            cmap=cmap_cells, 
                                            threshold=None,
                                            title_size2=title_size
                                            )
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()
                                    
                                    
                                
                                with ui.card(full_screen=False):
                                    ui.card_header("Scaled by net distance")
                                    @render.plot
                                    def plot4():
                                        figure = pu.migration_directions_with_kde_plus_mean(
                                            df=Track_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Cells', 
                                            scaling_metric='NET_DISTANCE', 
                                            cmap_normalization_metric=None, 
                                            cmap=cmap_cells, 
                                            threshold=None,
                                            title_size2=title_size2
                                            )
                                        return figure

                                    @render.download(label="Download", filename="Track directionality.png")
                                    def download2():
                                        figure = pu.migration_directions_with_kde_plus_mean(
                                            df=Track_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Cells', 
                                            scaling_metric='NET_DISTANCE', 
                                            cmap_normalization_metric=None, 
                                            cmap=cmap_cells, 
                                            threshold=None,
                                            title_size2=title_size
                                            )
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()
                            
                        with ui.card(full_screen=True):
                            ui.card_header("Migration heatmaps")
                            with ui.layout_column_wrap(width=1 / 2):
                                with ui.card(full_screen=False):
                                    ui.card_header("Standard")        
                                    @render.plot
                                    def plot5():
                                        return pu.df_gaussian_donut(
                                            df=Track_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Cells', 
                                            heatmap='inferno', 
                                            weight=None, 
                                            threshold=None,
                                            title_size2=title_size2,
                                            label_size=label_size,
                                            figtext_color=figtext_color,
                                            figtext_size=figtext_size
                                            )
                                    
                                    @render.download(label="Download", filename="Cell migration heatmap.png")
                                    def download3():
                                        figure = pu.df_gaussian_donut(
                                            df=Track_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Cells', 
                                            heatmap='inferno', 
                                            weight=None, 
                                            threshold=None,
                                            title_size2=title_size2,
                                            label_size=label_size,
                                            figtext_color=figtext_color,
                                            figtext_size=figtext_size
                                            )
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()

                                with ui.card(full_screen=False):
                                    ui.card_header("Weighted")
                                    with ui.value_box(
                                    full_screen=False,
                                    theme="text-red"
                                    ):
                                        ""
                                        "Currently unavailable"
                                        ""

                with ui.nav_panel("Whole dataset histograms"):
                    def histogram_cells_distance(df, metric, str):
                        # Sort the DataFrame by 'TRACK_LENGTH' in ascending order
                        df_sorted = df.sort_values(by=metric)

                        norm = mcolors.Normalize(vmin=df_sorted["NUM_FRAMES"].min(), vmax=df_sorted["NUM_FRAMES"].max())
                        cmap = plt.colormaps["ocean_r"]

                        # Create new artificial IDs for sorting purposes (1 for lowest distance, N for highest)
                        df_sorted["Artificial_ID"] = range(1, len(df_sorted) + 1)

                        x_span = PlotParams.x_span(df_sorted)

                        # Create the figure and axis for the plot
                        fig, ax = plt.subplots(figsize=(x_span, 8))
                        fig.set_tight_layout(True)
                        width = 6

                        # Loop through each row to plot each cell's data
                        for idx, row in df_sorted.iterrows():
                            artificial_id = row["Artificial_ID"]
                            track_length = row[metric]
                            num_frames = row["NUM_FRAMES"]

                            # Get the color based on the number of frames using the viridis colormap
                            line_color = cmap(norm(num_frames))

                            # Plot the "chimney" or vertical line
                            ax.vlines(
                                x=artificial_id,  # X position for the cell
                                ymin=track_length,  # Starting point of the line (y position)
                                ymax=track_length + num_frames,  # End point based on number of frames (height)
                                color=line_color,
                                linewidth=width,
                                )

                            plt.plot(artificial_id, track_length, '_', zorder=5, color="lavender")

                            # Add the mean number of frames as text above each chimney
                            ax.text(
                            artificial_id,  # X position (same as the chimney)
                            track_length + num_frames + 1,  # Y position (slightly above the chimney)
                            f"{round(num_frames)}",  # The text to display (formatted mean)
                            ha='center',  # Horizontal alignment center
                            va='bottom',  # Vertical alignment bottom
                            fontsize=6,  # Adjust font size if necessary
                            color='black',  # Color of the text
                            style='italic'  # Italicize the text
                            )

                            x = int(row['Artificial_ID'])

                            plt.xticks(range(x), rotation=90) # add loads of ticks
                            plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
                            plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)


                        max_y = df_sorted[metric].max()
                        num_x_values = df_sorted[metric].count()

                        # Adjust the plot aesthetics
                        plt.tick_params(axis='x', rotation=60)
                        plt.tick_params(axis='y', labelsize=8)
                        plt.xticks(range(num_x_values)) # add loads of ticks
                        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
                        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)

                        # Set ticks, labels and title
                        ax.set_xticks(range(1, num_x_values + 1))
                        ax.set_yticks(np.arange(0, max_y + 1, 10))
                        ax.set_xlabel(f"Cells (sorted by {str} distance)")
                        ax.set_ylabel(f"{str} distance traveled [μm]")
                        ax.set_title(f"{str} Distance Traveled by Cells\nWith Length Representing Number of Frames")

                        # Invert x-axis so the highest distance is on the left
                        plt.gca().invert_xaxis()

                        ax.set_xlim(right=0, left=num_x_values+1)  # Adjust the left limit as needed

                        # Show the plot
                        # plt.savefig(op.join(save_path, f"02f_Histogram_{str}_distance_traveled_per_cell.png"))
                        # plt.show()

                        return plt.gcf()
                    
                    with ui.layout_column_wrap(width=2 / 2):
                        with ui.card(full_screen=False): 
                            with ui.layout_columns(
                                col_widths=(12,12)
                            ): 
                                with ui.card(full_screen=True):
                                    ui.card_header("Net distances travelled")
                                    @render.plot(
                                            width=3600,
                                            height=500
                                            )
                                    def plot6():
                                        figure = histogram_cells_distance(
                                            df=Track_stats_df.get(), 
                                            metric='NET_DISTANCE', 
                                            str='Net'
                                            )
                                        return figure
                                    
                                    @render.download(label="Download", filename="Net distances travelled.png")
                                    def download11():
                                        figure = histogram_cells_distance(
                                            df=Track_stats_df.get(), 
                                            metric='NET_DISTANCE', 
                                            str='Net'
                                            )
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()

                                with ui.card(full_screen=True):
                                    ui.card_header("Track lengths")
                                    @render.plot(
                                            width=3800,
                                            height=1000
                                            )
                                    def plot12():
                                        figure = histogram_cells_distance(
                                            df=Track_stats_df.get(), 
                                            metric='TRACK_LENGTH', 
                                            str='Total'
                                            )
                                        return figure
                                    
                                    @render.download(
                                            label="Download", 
                                            filename="Track lengths.png"
                                            )
                                    def download12():
                                        figure = histogram_cells_distance(
                                            df=Track_stats_df.get(), 
                                            metric='TRACK_LENGTH', 
                                            str='Total'
                                            )
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()
                                    


                
        with ui.nav_panel("Frames"):
            
            with ui.navset_card_tab(id="tab2"):
                with ui.nav_panel("Histograms"):
                    with ui.layout_columns(
                        col_widths={"sm": (12,6,6)},
                        row_heights=(3,4),
                        # height="700px",
                    ):
                        
                        with ui.card(full_screen=True):
                            ui.card_header("Speed histogram")
                            @render.plot
                            def plot7():
                                figure = pu.histogram_frame_speed(df=Frame_stats_df.get())
                                return figure

                            @render.download(label="Download", filename="Track directionality.png")
                            def download4():
                                figure = pu.histogram_frame_speed(df=Frame_stats_df.get())
                                with io.BytesIO() as buf:
                                    figure.savefig(buf, format="png", dpi=300)
                                    yield buf.getvalue()

                with ui.nav_panel("Directionality plots"):
                    with ui.layout_columns():
                        with ui.card(full_screen=True):
                            ui.card_header("Directionality")
                            with ui.layout_column_wrap(width=1 / 2):
                                with ui.card(full_screen=False):
                                    ui.card_header("Standard - Scaled by mean distance")
                                    @render.plot
                                    def plot8():
                                        return pu.migration_directions_with_kde_plus_mean(
                                            df=Frame_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Frames (weighted)', 
                                            scaling_metric='MEAN_DISTANCE', 
                                            cmap_normalization_metric='POSITION_T', 
                                            cmap=cmap_frames, 
                                            threshold=None,
                                            title_size2=title_size2
                                            )
                                with ui.card(full_screen=False):
                                    ui.card_header("Weighted - Scaled by mean distance")
                                    @render.plot
                                    def plot9():
                                        return pu.migration_directions_with_kde_plus_mean(
                                            df=Frame_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
                                            subject='Frames (weighted)', 
                                            scaling_metric='MEAN_DISTANCE', 
                                            cmap_normalization_metric='POSITION_T', 
                                            cmap=cmap_frames, 
                                            threshold=None,
                                            title_size2=title_size2
                                            )
                
                        with ui.card(full_screen=True):
                            ui.card_header("Migration heatmaps")
                            with ui.layout_column_wrap(width=1 / 2):
                                with ui.card(full_screen=False):
                                    ui.card_header("Standard")        
                                    @render.plot
                                    def plot10():
                                        return pu.df_gaussian_donut(
                                            df=Frame_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Frames', 
                                            heatmap='viridis', 
                                            weight=None, 
                                            threshold=None,
                                            title_size2=title_size2,
                                            label_size=label_size,
                                            figtext_color=figtext_color,
                                            figtext_size=figtext_size
                                            )
                                with ui.card(full_screen=False):
                                    ui.card_header("Weighted")
                                    @render.plot
                                    def plot11():
                                        return pu.df_gaussian_donut(
                                            df=Frame_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
                                            subject='Frames', 
                                            heatmap='viridis', 
                                            weight='mean distance traveled', 
                                            threshold=None,
                                            title_size2=title_size2,
                                            label_size=label_size,
                                            figtext_color=figtext_color,
                                            figtext_size=figtext_size
                                            )


# ===========================================================================================================================================================================================================================================================================
# Frame panel
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# with ui.nav_panel("Stats"):


    # ===========================================================================================================================================================================================================================================================================
    # Plotting functions
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




ui.nav_spacer()  
with ui.nav_control():  
    ui.input_dark_mode(mode="light")