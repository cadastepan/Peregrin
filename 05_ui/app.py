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
# Creating reactive values for thresholding the data

Spot_stats_df_T1 = reactive.value()
Spot_stats_df_T2 = reactive.value()
Spot_stats_df_T3 = reactive.value()
Track_stats_df_T1 = reactive.value()
Track_stats_df_T2 = reactive.value()
Track_stats_df_T3 = reactive.value()


# ===========================================================================================================================================================================================================================================================================
# Creating reactive variables for processed dataframe storage

Buttered_df = reactive.value()
Spot_stats_df = reactive.value()
Track_stats_df = reactive.value()
Frame_stats_df = reactive.value()


# ===========================================================================================================================================================================================================================================================================
# Creating other reactive variables 

slider_valuesT1 = reactive.value()   # Creating a rective value for the slider values
slider_valuesT2 = reactive.value()   # Creating a rective value for the slider values
slider_valuesT3 = reactive.value()   # Creating a rective value for the slider values
slider_valuesT4 = reactive.value()   # Creating a rective value for the slider values
Track_metrics = reactive.value()     # Creating a reactive value for the track metrics
Spot_metrics = reactive.value()      # Creating a reactive value for the spot metrics

count = reactive.value(1)            # Data input counter

conditions = reactive.value()        # Creating a reactive value for the conditions



Thresholding_metrics ={
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
}

Thresholding_filters = {
    "literal": "Literal",
    "percentile": "Percentile",
}

# ===========================================================================================================================================================================================================================================================================
# Data panel
# ===========================================================================================================================================================================================================================================================================
# Reading a selected CSV file and cleaning it
# Extracting separate dataframes











# Directionality metric 
# When downloading, I could make it possible to download the merged df and the separate datasets as well, in which case I would exclude the CONDITION column but rather include it in the name of the file
# I may include some metadata as well for download such as the data of the analysis and what not idk
# # I could also make it possible to download the data as a .txt file or .xlsx file
# # # 2D visualization - gating
# # # remove the percentile thresholding opptions
# # # make the thresholding the way that inputs are, so that you can add and remove them
# # # maybe an option in 1D/2D thresholding











with ui.nav_panel("Input"):
    
    with ui.div(id="data-inputs"): # div container for flow content

        # =============================================================================================================================================================================================================================================================================
        # Buttons for adding and removing additional data input

        ui.input_action_button("more", "Add data input")
        ui.input_action_button("less", "Remove data input")


        # =============================================================================================================================================================================================================================================================================
        # Default data input slot

        @render.ui
        def default_input():
            default_browser = ui.input_file("file1", "Input CSV", accept=[".csv"], multiple=True, placeholder="No files selected")
            default_label = ui.input_text("label1", "Condition 0", placeholder="write something")
            return default_label, default_browser


        # =============================================================================================================================================================================================================================================================================
        # Additional data input slots - reacting on the buttons

        @reactive.effect
        @reactive.event(input.more)                             # "Add data input" button sensor
        def add_inputs():
            if input.more():                                    # REACTION:
                count.set(count.get() + 1)                      # Increasing the input count
                adding = count.get()                            # Getting the current input count

                browser = ui.input_file(                        # CSV file browser
                    id=f"file_{adding}", 
                    label=f"Input CSV {adding}", 
                    accept=[".csv"], 
                    multiple=True, 
                    placeholder="No files selected"
                    )
                label = ui.input_text(                          # Data labeling text window
                    id=f"label_{adding}", 
                    label=f"Condition {adding}", 
                    placeholder="write something"
                    )

                ui.insert_ui(                                   # Rendering the additional input slot container
                    ui.div(                                     # container consisting of the label, browser and use button
                        {"id": f"additional-input-{adding}"}, 
                        label, browser),
                        selector="#data-inputs",
                        where="beforeEnd",
                )

        @reactive.effect
        @reactive.event(input.less)                             # "Remove data input" button sensor
        def remove_inputs():
            if input.less():                                    # REACTION:
                removing = count.get()                          # Getting the current input count
                ui.remove_ui(f"#additional-input-{removing}")   # Removing the last input slot (one with the current input count)
                if count.get() > 1:                             # Decreasing the input count
                    count.set(removing - 1)                     
                else:
                    pass


    @reactive.calc 
    def parsed_file():                                                          # File-reading 
        
        # =============================================================================================================================================================================================================================================================================
        # Processing the default input files

                       
        inpt_file_list_dflt: list[FileInfo] | None = input.file1()                      # Getting the list of default input files

        if inpt_file_list_dflt is None:
            default = pd.DataFrame()
        
        else:
            all_data_dflt = []
            for file_count, file_dflt in enumerate(inpt_file_list_dflt, start=1):       # Enumerate and cycle through default input files
                df_dflt = pd.read_csv(file_dflt["datapath"])                     
                buttered_dflt = du.butter(df_dflt)                 # Process the DataFrame


                                                                    
                label_dflt = input.label1()                                             # Getting the label to assign the 'CONDITION' column parameter
                if not label_dflt or label_dflt is None:                                # If no label is provided, assign a default one
                    buttered_dflt['CONDITION'] = f"file_{file_count}"
                else:                                                                   # Else, assign the given lable
                    buttered_dflt['CONDITION'] = f"{label_dflt} {file_count}"

                buttered_dflt = buttered_dflt.drop_duplicates()                          # Drop duplicates
                all_data_dflt += [buttered_dflt]                                     # Store processed DataFrame

                default = pd.concat(all_data_dflt, axis=0)            # Join the DataFrames
                
        # =============================================================================================================================================================================================================================================================================
        # Processing the additional input files

        browse_count = count.get()                                                      # Getting the current additional input slot count

        for i in range(2, browse_count + 1):                                            # Cycle trough the additional input slots 

            inpt_file_list_addtnl: list[FileInfo] | None = input[f"file_{i}"]()         # Getting the list of files

            if inpt_file_list_addtnl is None:
                additional = pd.DataFrame()
            
            else:
                all_data_addtnl = []                                                      # List storing processed DataFrames                            
                for file_addtnl in inpt_file_list_addtnl:                               # Enumerate and cycle through additional input files
                    df_addtnl = pd.read_csv(file_addtnl["datapath"])                  
                    buttered_addtnl = du.butter(df_addtnl)

                    label_addtnl = input[f"label_{i}"]()                                # Getting the label to assign the 'CONDITION' column parameter
                    if not label_addtnl or label_addtnl is None:                        # If no label is provided, assign a default one
                        buttered_addtnl['CONDITION'] = f"file_{i}"
                    else:                                                               # Else, assign the given lable
                        buttered_addtnl['CONDITION'] = f"{label_addtnl} {i}"

                    buttered_addtnl = buttered_addtnl               # Drop duplicates
                    all_data_addtnl += [buttered_addtnl]                                  # Store processed DataFrame
                    

                    additional = pd.concat(all_data_addtnl, axis=0)          # Join the DataFrames

        # =============================================================================================================================================================================================================================================================================
        # Merging the default and additional input files

        if browse_count == 1:
            return pd.DataFrame(default)
        elif additional.empty:
            return pd.DataFrame(default)
        else:
            return pd.DataFrame(pd.concat([default, additional], axis=0))



                
            

            
 



with ui.nav_panel("Data frames"):  # Data panel

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

        distances_for_each_cell_per_frame_df = du.calculate_traveled_distances_for_each_cell_per_frame(buttered)        # Call the function to calculate distances for each cell per frame and create the Spot_statistics .csv file
        direction_for_each_cell_per_frame_df = du.calculate_direction_of_travel_for_each_cell_per_frame(buttered)       # Call the function to calculate direction_for_each_cell_per_frame_df

        Spot_stats_dfs = [buttered, distances_for_each_cell_per_frame_df, direction_for_each_cell_per_frame_df]
        Spot_stats = du.merge_dfs(Spot_stats_dfs, on=['CONDITION', 'TRACK_ID', 'POSITION_T']) # Merge the dataframes
        Spot_stats = Spot_stats.sort_values(by=['CONDITION','TRACK_ID', 'POSITION_T'])	

        return Spot_stats


    @reactive.effect
    def update_Spot_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        else:
            Spot_stats = process_spot_data()
            raw_Spot_stats_df.set(Spot_stats)
            Spot_metrics.set(Spot_stats.columns)



    @reactive.calc
    def process_track_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()

        Spot_stats = raw_Spot_stats_df.get()

        if Spot_stats.empty:
            return pd.DataFrame()

        tracks_lengths_and_net_distances_df = du.calculate_track_lengths_and_net_distances(Spot_stats) # Calling function to calculate the total distance traveled for each cell from the distances_for_each_cell_per_frame_df
        confinement_ratios_df = du.calculate_confinement_ratio_for_each_cell(tracks_lengths_and_net_distances_df) # Call the function to calculate confinement ratios from the Track_statistics1_df and write it into the Track_statistics1_df
        track_directions_df = du.calculate_absolute_directions_per_cell(Spot_stats) # Call the function to calculate directions_per_cell_df
        frames_per_track = du.calculate_number_of_frames_per_cell(Spot_stats)
        speeds_per_cell = du.calculate_speed(Spot_stats, 'TRACK_ID')

        Track_stats_dfs = [tracks_lengths_and_net_distances_df, confinement_ratios_df, track_directions_df, frames_per_track, speeds_per_cell]
        Track_stats = du.merge_dfs(Track_stats_dfs, on=['CONDITION','TRACK_ID'])

        Track_stats = Track_stats.sort_values(by=['CONDITION','TRACK_ID'])

        return Track_stats

    
    @reactive.calc
    def process_frame_data():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()

        Spot_stats = Spot_stats_df.get()

        if Spot_stats.empty:
            return pd.DataFrame()
        
        distances_per_frame_df = du.calculate_distances_per_frame(Spot_stats) # Call the function to calculate distances_per_frame_df
        absolute_directions_per_frame_df = du.calculate_absolute_directions_per_frame(Spot_stats) # Call the function to calculate directions_per_frame_df
        speeds_per_frame = du.calculate_speed(Spot_stats, 'POSITION_T') # Call the function to calculate speeds_per_frame

        Frame_stats_dfs = [distances_per_frame_df, absolute_directions_per_frame_df, speeds_per_frame]

        Frame_stats = du.merge_dfs(Frame_stats_dfs, on='POSITION_T')
        Frame_stats = Frame_stats.merge(Spot_stats['POSITION_T'].drop_duplicates(), on='POSITION_T')

        Frame_stats = Frame_stats.sort_values(by=['CONDITION','POSITION_T'])

        return Frame_stats
    

    @reactive.effect
    def update_Track_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        else:
            Track_stats = process_track_data()
            raw_Track_stats_df.set(Track_stats)
            Track_metrics.set(Track_stats.columns)

    @reactive.effect
    def update_Frame_stats_df():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        else:
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
                else:
                    Spot_stats = Spot_stats_df.get()
                    return render.DataGrid(Spot_stats)
                
            @render.download(label="Download", filename="Spot_stats.csv")
            def download_spot_stats():
                with io.BytesIO() as buf:
                    Spot_stats_df.get().to_csv(buf, index=False)
                    yield buf.getvalue()
            
        
        with ui.card():
            ui.card_header("Track stats")
            
            @render.data_frame
            def render_track_stats():
                file: list[FileInfo] | None = input.file1()
                if file is None:
                    return pd.DataFrame()
                else:
                    Track_stats = Track_stats_df.get()
                    return render.DataGrid(Track_stats)
                
            @render.download(label="Download", filename="Track_stats.csv")
            def download_track_stats():
                with io.BytesIO() as buf:
                    Track_stats_df.get().to_csv(buf, index=False)
                    yield buf.getvalue()
            
            
        with ui.card():
            ui.card_header("Frame stats")

            @render.data_frame
            def render_frame_stats():
                file: list[FileInfo] | None = input.file1()
                if file is None:
                    return pd.DataFrame()
                else:
                    Frame_stats = Frame_stats_df.get()
                    return render.DataGrid(Frame_stats)
                
            @render.download(label="Download", filename="Frame_stats.csv")
            def download_frame_stats():
                with io.BytesIO() as buf:
                    Frame_stats_df.get().to_csv(buf, index=False)
                    yield buf.getvalue()
    


















# ===========================================================================================================================================================================================================================================================================
# Thresholding panel functions


def update_slider(filter_type, slider, slider_values):
    if filter_type == "percentile":
        ui.update_slider(id=slider, min=0, max=100, value=(0, 100), step=1)
    elif filter_type == "literal":
        values = slider_values.get()
        range = values[1] - values[0]

        if range <= 10:
            steps = 0.01
        elif range <= 100:
            steps = 0.1
        else:
            steps = 1
        
        if values:
            ui.update_slider(id=slider, min=values[0], max=values[1], value=values, step=steps)

def update_slider_values(metric, filter, dfA, dfB, slider_values):
    if metric in Track_metrics.get():
        try:
            if filter == "literal":
                if dfA.empty:
                    slider_values.set([0, 100])
                else:
                    values = du.values_for_a_metric(dfA, metric)
                    slider_values.set(values)
            elif filter == "percentile":
                slider_values.set([0, 100])
        except Exception as e:
            slider_values.set([0, 100])
    elif metric in Spot_metrics.get():
        try:
            if filter == "literal":
                if dfB.empty:
                    slider_values.set([0, 100])
                else:
                    values = du.values_for_a_metric(dfB, metric)
                    slider_values.set(values)
            elif filter == "percentile":
                slider_values.set([0, 100])
        except Exception as e:
            slider_values.set([0, 100])
    
def thresholded_histogram(metric, filter_type, slider_range, dfA, dfB):
    try:
        if metric in Track_metrics.get():
            data = dfA.get()
        elif metric in Spot_metrics.get():
            data = dfB.get()
        elif data.empty:
            return plt.figure()
        else:
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
        ax.spines[['top','left','right']].set_visible(False)

        return fig
    
    except ValueError:
        return plt.figure()

def data_thresholding_numbers(df):
    raw = raw_Track_stats_df.get().shape[0]
    filtered = df.get().shape[0]
    filtered_out = raw - filtered

    # Filtered data in percents
    filtered_prcbt = filtered / raw * 100
    filtered_out_prcbt = filtered_out / raw * 100

    return f"Cells in total: {raw}", f"In focus: {round(filtered_prcbt)} % ({filtered})", f"Filtered out: {round(filtered_out_prcbt)} % ({filtered_out})"

def thresholded_data(filter_type, metric, slider_range, dfA, dfB):
    if filter_type == "percentile":
        if metric in Track_metrics.get():
            return du.percentile_thresholding(dfA, metric, slider_range)
        elif metric in Spot_metrics.get():
            return du.percentile_thresholding(dfB, metric, slider_range)
    elif filter_type == "literal":
        if metric in Track_metrics.get():
            return du.literal_thresholding(dfA, metric, slider_range)
        elif metric in Spot_metrics.get():
            return du.literal_thresholding(dfB, metric, slider_range)

def update_thresholded_data(metric, dfA, dfB, df0A, df0B, thresholded_df):
    if metric in Track_metrics.get():
        dfA.set(thresholded_df)
        dfB.set(du.dataframe_filter(df0B.get(), dfA.get()))
    elif metric in Spot_metrics.get():
        dfB.set(thresholded_df)
        dfA.set(du.dataframe_filter(df0A.get(), dfB.get()))

def make_panel():
    try:
        return ui.accordion_panel(
            title="Section",
            content=("Content of the new section")
        )
    except AttributeError:
        pass


# ===========================================================================================================================================================================================================================================================================
# Sidebar 

with ui.sidebar(open="open", position="right", bg="f8f8f8"): 


    # ===========================================================================================================================================================================================================================================================================
    # Condition selection for visualizing
    ui.input_select(
        "condition",
        "Select a condition to be vizualized",
        choices=[]
    )

    @reactive.effect
    def update_selection():
        ui.update_select(
            id="condition",
            label="Select a condition to be vizualized",
            choices=conditions.get(),
        )



    # ===========================================================================================================================================================================================================================================================================
    # Thresholding

    with ui.accordion(id="sidebar_acc"):


        # ===========================================================================================================================================================================================================================================================================
        # Thresholding 1 panel

        with ui.accordion_panel(title="Tresholding"):


            # ===========================================================================================================================================================================================================================================================================
            # Creating a possibility for thresholding metric selection
            # Creating a possibility for thresholding filter selection
            # Creating a slider for thresholding

            ui.input_select(  
                "metricA",  
                "Thresholding metric:",  
                Thresholding_metrics 
            )  

            ui.input_select(
                "filterA",
                "Thresholding filter:",
                Thresholding_filters
            )

            ui.input_slider(
                "sliderA",
                "Threshold",
                min=0,
                max=100,
                value=(0, 100)
            )


            # ===========================================================================================================================================================================================================================================================================
            # Reactive functions updating the slider values
            
            @reactive.effect
            def update_sliderA():
                return update_slider(input.filterA(), "sliderA", slider_valuesT1)

            @reactive.effect
            def update_slider_valuesA():
                return update_slider_values(input.metricA(), input.filterA(), raw_Track_stats_df.get(), raw_Spot_stats_df.get(), slider_valuesT1)


            # ===========================================================================================================================================================================================================================================================================
            # Thresholding the data based on percentiles
            
            @reactive.calc
            def thresholded_dataA():
                return thresholded_data(input.filterA(), input.metricA(), input.sliderA(), raw_Track_stats_df.get(), raw_Spot_stats_df.get())

            @reactive.effect
            def update_thresholded_dataA():
                return update_thresholded_data(input.metricA(), Track_stats_df_T1, Spot_stats_df_T1, raw_Track_stats_df, raw_Spot_stats_df, thresholded_dataA())

            @render.plot
            def threshold_histogramA():
                return thresholded_histogram(input.metricA(), input.filterA(), input.sliderA(), raw_Track_stats_df, raw_Spot_stats_df)

            @render.text
            def data_thresholding_numbersA1():
                a, b, c = data_thresholding_numbers(Track_stats_df_T1)
                return a

            @render.text
            def data_thresholding_numbersA2():
                a, b, c = data_thresholding_numbers(Track_stats_df_T1)
                return b

            @render.text
            def data_thresholding_numbersA3():
                a, b, c = data_thresholding_numbers(Track_stats_df_T1)
                return c
            

        # ===========================================================================================================================================================================================================================================================================
        # Thresholding 2 panel

        with ui.accordion_panel(title="Tresholding 2"):


            # ===========================================================================================================================================================================================================================================================================
            # Creating a possibility for thresholding metric selection
            # Creating a possibility for thresholding filter selection
            # Creating a slider for thresholding

            ui.input_select(  
                "metricB",  
                "Thresholding metric:",  
                Thresholding_metrics 
            )  

            ui.input_select(
                "filterB",
                "Thresholding filter:",
                Thresholding_filters
            )

            ui.input_slider(
                "sliderB",
                "Threshold",
                min=0,
                max=100,
                value=(0, 100)
            )


            # ===========================================================================================================================================================================================================================================================================
            # Reactive functions updating the slider values
            
            @reactive.effect
            def update_sliderB():
                return update_slider(input.filterB(), "sliderB", slider_valuesT2)

            @reactive.effect
            def update_slider_valuesB():
                return update_slider_values(input.metricB(), input.filterB(), Track_stats_df_T1.get(), Spot_stats_df_T1.get(), slider_valuesT2)


            # ===========================================================================================================================================================================================================================================================================
            # Thresholding the data based on percentiles
            
            @reactive.calc
            def thresholded_dataB():
                return thresholded_data(input.filterB(), input.metricB(), input.sliderB(), Track_stats_df_T1.get(), Spot_stats_df_T1.get())

            @reactive.effect
            def update_thresholded_dataB():
                return update_thresholded_data(input.metricB(), Track_stats_df_T2, Spot_stats_df_T2, Track_stats_df_T1, Spot_stats_df_T1, thresholded_dataB())

            @render.plot
            def threshold_histogramB():
                return thresholded_histogram(input.metricB(), input.filterB(), input.sliderB(), Track_stats_df_T1, Spot_stats_df_T1)

            @render.text
            def data_thresholding_numbersB1():
                a, b, c = data_thresholding_numbers(Track_stats_df_T2)
                return a

            @render.text
            def data_thresholding_numbersB2():
                a, b, c = data_thresholding_numbers(Track_stats_df_T2)
                return b

            @render.text
            def data_thresholding_numbersB3():
                a, b, c = data_thresholding_numbers(Track_stats_df_T2)
                return c
            

        # ===========================================================================================================================================================================================================================================================================
        # Thresholding 3 panel

        with ui.accordion_panel(title="Tresholding 3"):
            
            
            # ===========================================================================================================================================================================================================================================================================
            # Creating a possibility for thresholding metric selection
            # Creating a possibility for thresholding filter selection
            # Creating a slider for thresholding

            ui.input_select(  
                "metricC",  
                "Thresholding metric:",  
                Thresholding_metrics 
            )  

            ui.input_select(
                "filterC",
                "Thresholding filter:",
                Thresholding_filters
            )

            ui.input_slider(
                "sliderC",
                "Threshold",
                min=0,
                max=100,
                value=(0, 100)
            )


            # ===========================================================================================================================================================================================================================================================================
            # Reactive functions updating the slider values
            
            @reactive.effect
            def update_sliderC():
                return update_slider(input.filterC(), "sliderC", slider_valuesT3)

            @reactive.effect
            def update_slider_valuesC():
                return update_slider_values(input.metricC(), input.filterC(), Track_stats_df_T2.get(), Spot_stats_df_T2.get(), slider_valuesT3)


            # ===========================================================================================================================================================================================================================================================================
            # Thresholding the data based on percentiles
            
            @reactive.calc
            def thresholded_dataC():
                return thresholded_data(input.filterC(), input.metricC(), input.sliderC(), Track_stats_df_T2.get(), Spot_stats_df_T2.get())

            @reactive.effect
            def update_thresholded_dataC():
                return update_thresholded_data(input.metricC(), Track_stats_df_T3, Spot_stats_df_T3, Track_stats_df_T2, Spot_stats_df_T2, thresholded_dataC())

            @render.plot
            def threshold_histogramC():
                return thresholded_histogram(input.metricC(), input.filterC(), input.sliderC(), Track_stats_df_T2, Spot_stats_df_T2)

            @render.text
            def data_thresholding_numbersC1():
                a, b, c = data_thresholding_numbers(Track_stats_df_T3)
                return a

            @render.text
            def data_thresholding_numbersC2():
                a, b, c = data_thresholding_numbers(Track_stats_df_T3)
                return b

            @render.text
            def data_thresholding_numbersC3():
                a, b, c = data_thresholding_numbers(Track_stats_df_T3)
                return c
    

        # ===========================================================================================================================================================================================================================================================================
        # Thresholding 4 panel

        with ui.accordion_panel(title="Tresholding 4"):
            
            
            # ===========================================================================================================================================================================================================================================================================
            # Creating a possibility for thresholding metric selection
            # Creating a possibility for thresholding filter selection
            # Creating a slider for thresholding

            ui.input_select(  
                "metricD",  
                "Thresholding metric:",  
                Thresholding_metrics 
            )  

            ui.input_select(
                "filterD",
                "Thresholding filter:",
                Thresholding_filters
            )

            ui.input_slider(
                "sliderD",
                "Threshold",
                min=0,
                max=100,
                value=(0, 100)
            )


            # ===========================================================================================================================================================================================================================================================================
            # Reactive functions updating the slider values
            
            @reactive.effect
            def update_sliderD():
                return update_slider(input.filterD(), "sliderD", slider_valuesT4)

            @reactive.effect
            def update_slider_valuesD():
                return update_slider_values(input.metricD(), input.filterD(), Track_stats_df_T3.get(), Spot_stats_df_T3.get(), slider_valuesT4)


            # ===========================================================================================================================================================================================================================================================================
            # Thresholding the data based on percentiles
            
            @reactive.calc
            def thresholded_dataD():
                return thresholded_data(input.filterD(), input.metricD(), input.sliderD(), Track_stats_df_T3.get(), Spot_stats_df_T3.get())

            @reactive.effect
            def update_thresholded_dataD():
                return update_thresholded_data(input.metricD(), Track_stats_df, Spot_stats_df, Track_stats_df_T3, Spot_stats_df_T3, thresholded_dataD())

            @render.plot
            def threshold_histogramD():
                return thresholded_histogram(input.metricD(), input.filterD(), input.sliderD(), Track_stats_df_T3, Spot_stats_df_T3)

            @render.text
            def data_thresholding_numbersD1():
                a, b, c = data_thresholding_numbers(Track_stats_df)
                return a

            @render.text
            def data_thresholding_numbersD2():
                a, b, c = data_thresholding_numbers(Track_stats_df)
                return b

            @render.text
            def data_thresholding_numbersD3():
                a, b, c = data_thresholding_numbers(Track_stats_df)
                return c
    
            















































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
                            def raw_tracks():
                                return pu.visualize_full_tracks(
                                    df=Spot_stats_df.get(), 
                                    df2=Track_stats_df.get(), 
                                    threshold=None, 
                                    lw=0.5
                                    )

                            @render.download(label="Download", filename="Raw tracks visualization.png")
                            def download_raw_tracks():
                                figure = pu.visualize_full_tracks(
                                    df=Spot_stats_df.get(), 
                                    df2=Track_stats_df.get(), 
                                    threshold=None, 
                                    lw=0.5
                                    )
                                with io.BytesIO() as buf:
                                    figure.savefig(buf, format="png", dpi=300)
                                    yield buf.getvalue()

                        with ui.card(full_screen=True):
                            ui.card_header("Smoothened tracks visualization")
                            @render.plot
                            def smoothened_tracks():
                                return pu.visualize_smoothened_tracks(
                                    df=Spot_stats_df.get(), 
                                    df2=Track_stats_df.get(), 
                                    threshold=None, 
                                    smoothing_type='moving_average', 
                                    smoothing_index=50, 
                                    lw=0.8
                                    )

                            @render.download(label="Download", filename="Smoothened tracks visualization.png")
                            def download_smoothened_tracks():
                                figure = pu.visualize_smoothened_tracks(
                                    df=Spot_stats_df.get(), 
                                    df2=Track_stats_df.get(), 
                                    threshold=None, 
                                    smoothing_type='moving_average', 
                                    smoothing_index=50, 
                                    lw=0.8
                                    )
                                with io.BytesIO() as buf:
                                    figure.savefig(buf, format="png", dpi=300)
                                    yield buf.getvalue()

                with ui.nav_panel("Directionality plots"):
                    with ui.layout_columns():
                        with ui.card(full_screen=True):  
                            ui.card_header("Directionality")
                            with ui.layout_column_wrap(width=1 / 2):
                                with ui.card(full_screen=False):
                                    ui.card_header("Scaled by confinement ratio")
                                    @render.plot
                                    def migration_direction_tracks1():
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
                                    
                                    @render.download(label="Download", filename="Track directionality (scaled by confinement ratio).png")
                                    def download_migration_direction_tracks1():
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
                                    def migration_direction_tracks2():
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

                                    @render.download(label="Download", filename="Track directionality (scaled by net distance).png")
                                    def download_migration_direction_tracks2():
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
                                    def tracks_migration_heatmap():
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
                                    def download_tracks_migration_heatmap():
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
                                    def cell_histogram_1():
                                        figure = pu.histogram_cells_distance(
                                            df=Track_stats_df.get(), 
                                            metric='NET_DISTANCE', 
                                            str='Net'
                                            )
                                        return figure
                                    
                                    @render.download(label="Download", filename="Net distances travelled.png")
                                    def download_cell_histogram_1():
                                        figure = pu.histogram_cells_distance(
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
                                    def cell_histogram_2():
                                        figure = pu.histogram_cells_distance(
                                            df=Track_stats_df.get(), 
                                            metric='TRACK_LENGTH', 
                                            str='Total'
                                            )
                                        return figure
                                    
                                    @render.download(label="Download", filename="Track lengths.png")
                                    def download_cell_histogram_2():
                                        figure = pu.histogram_cells_distance(
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
                            def migration_histogram():
                                figure = pu.histogram_frame_speed(df=Frame_stats_df.get())
                                return figure

                            @render.download(label="Download", filename="Speed histogram.png")
                            def download_migration_histogram():
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
                                    def migration_direction_frames1():
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
                                    
                                    @render.download(label="Download", filename="Frame directionality (standard - scaled by mean distance).png")
                                    def download_migration_direction_frames1():
                                        figure = pu.migration_directions_with_kde_plus_mean(
                                            df=Frame_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD', 
                                            subject='Frames (weighted)', 
                                            scaling_metric='MEAN_DISTANCE', 
                                            cmap_normalization_metric='POSITION_T', 
                                            cmap=cmap_frames, 
                                            threshold=None,
                                            title_size2=title_size2
                                            )
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()

                                with ui.card(full_screen=False):
                                    ui.card_header("Weighted - Scaled by mean distance")
                                    @render.plot
                                    def migration_direction_frames2():
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
                                    
                                    @render.download(label="Download", filename="Frame directionality (weighted - scaled by mean distance).png")
                                    def download_migration_direction_frames2():
                                        figure = pu.migration_directions_with_kde_plus_mean(
                                            df=Frame_stats_df.get(), 
                                            metric='MEAN_DIRECTION_RAD_weight_mean_dis', 
                                            subject='Frames (weighted)', 
                                            scaling_metric='MEAN_DISTANCE', 
                                            cmap_normalization_metric='POSITION_T', 
                                            cmap=cmap_frames, 
                                            threshold=None,
                                            title_size2=title_size2
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
                                    def frame_migration_heatmap_1():
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
                                    
                                    @render.download(label="Download", filename="Frame migration heatmap (standard).png")
                                    def download_frame_migration_heatmap_1():
                                        figure = pu.df_gaussian_donut(
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
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()

                                with ui.card(full_screen=False):
                                    ui.card_header("Weighted")
                                    @render.plot
                                    def frame_migration_heatmap_2():
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
                                    
                                    @render.download(label="Download", filename="Frame migration heatmap (weighted).png")
                                    def download_frame_migration_heatmap_2():
                                        figure = pu.df_gaussian_donut(
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
                                        with io.BytesIO() as buf:
                                            figure.savefig(buf, format="png", dpi=300)
                                            yield buf.getvalue()


# ===========================================================================================================================================================================================================================================================================
# Frame panel
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


ui.nav_spacer()  
with ui.nav_control():  
    ui.input_dark_mode(mode="light")




# ===========================================================================================================================================================================================================================================================================
# Action buttons for additional browse windows used for inputting other data frames and also making a window for each input which will leave a mark on the dataframe e.g. Treatment CK12 - which will be also written into a column specifying the conditions 
# Merging the dataframes
# exporting downloads in form of a rasterized file