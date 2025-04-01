import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from peregrin.scripts import PlotParams
import seaborn as sns
from itertools import combinations


def histogram_frame_speed(df):
    frames = df['POSITION_T'][1:-1]
    mean_speed = df['SPEED_MEAN'][1:-1]
    median_speed = df['SPEED_MEDIAN'][1:-1]

    # Apply Savitzky-Golay filter for smoothing
    mean_speed_smooth = savgol_filter(mean_speed, window_length=11, polyorder=1)
    median_speed_smooth = savgol_filter(median_speed, window_length=11, polyorder=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(frames, mean_speed, '.', label='Mean Speed', alpha=0.5)
    plt.plot(frames, median_speed, '.', label='Median Speed', alpha=0.5)
    plt.plot(frames, mean_speed_smooth, '-', label='Smoothed Mean Speed', linewidth=2)
    plt.plot(frames, median_speed_smooth, '-', label='Smoothed Median Speed', linewidth=2)

    # Set x-axis to start at 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.xlabel(r'Time $\it{[min]}$')
    plt.ylabel(r'Speed $\it{[μm]}$')
    plt.title('Mean and Median Speed per Frame')
    plt.legend()
    plt.grid(True)
    # plt.show()
    return plt.gcf()


def migration_directions_with_kde_plus_mean(df, metric, subject, scaling_metric, cmap_normalization_metric, cmap, threshold, title_size2):	

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
    ax.set_title(f'Mean Direction of Travel\nwith Kernel Density Estimate\n$\it{{{subject}}}$', fontsize=title_size2)
    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticklabels([])  # Remove angular labels

    return plt.gcf()

def donut(df, ax, outer_radius, inner_radius, kde_bw):
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

    return theta_mesh, r_mesh, kde_values, norm

def df_gaussian_donut(df, metric, subject, heatmap, weight, threshold, title_size2, label_size, figtext_color, figtext_size):

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
    ax.set_title(f'Heatmap of Migration Direction\n({subject})', pad=20, ha='center', fontsize=title_size2)
    
    # Add a colorbar
    cbar = plt.colorbar(ax.pcolormesh(theta_mesh, r_mesh, kde_values, shading='gouraud', cmap=heatmap, norm=norm), ax=ax, fraction=0.04, orientation='horizontal', pad=0.1)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)  # Remove outline
    
    # Add min and max labels below the colorbar
    cbar.ax.text(0.05, -0.4, 'min', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.95, -0.4, 'max', va='center', ha='center', color='black', transform=cbar.ax.transAxes, fontsize=9)

    # Add the density label below the min and max labels
    cbar.set_label('Density', labelpad=10, fontsize=label_size)
    
    # if weight == None:
    #     plt.savefig(f'04a_Plot_donut_heatmap-migration_direction_{subject}{threshold}.png', dpi=300)
    # else:
    #     weight = 'weighted by' + weight
    #     plt.figtext(0.515, 0.01, f'{weight}', ha='center', color=figtext_color, fontsize=figtext_size)
    #     plt.savefig(f'04a_Plot_donut_heatmap-migration_direction_{subject}{weight}{threshold}.png', dpi=300)

    # plt.show()

    return plt.gcf()

    # try to normalize the heatmap colors to the absolute 0 (not min of the kde values) and to the max of the kde values


def track_visuals(df2, c_mode, grid, lut_metric, title_size=12):
    
    fig, ax = plt.subplots(figsize=(13, 10))

    unique_tracks = df2[['CONDITION', 'REPLICATE', 'TRACK_ID', lut_metric]].drop_duplicates()

    lut_norm_df = df2[['TRACK_ID', lut_metric]].drop_duplicates()

    # Normalize the NET_DISTANCE to a 0-1 range
    lut_min = lut_norm_df[lut_metric].min()
    lut_max = lut_norm_df[lut_metric].max()
    norm = plt.Normalize(vmin=lut_min, vmax=lut_max)
    if c_mode == 'greyscale':
        colormap = plt.cm.gist_yarg
        grid_color = 'gainsboro'
        face_color = 'None'
        grid_alpha = 0.5
        if grid:
            grid_lines = '-.'
        else:
            grid_lines = 'None'
    else:
        if c_mode == 'color1':
            colormap = plt.cm.jet
        elif c_mode == 'color2':
            colormap = plt.cm.brg
        elif c_mode == 'color3':
            colormap = plt.cm.hot
        elif c_mode == 'color4':
            colormap = plt.cm.gnuplot
        elif c_mode == 'color5':
            colormap = plt.cm.viridis
        elif c_mode == 'color6':
            colormap = plt.cm.rainbow
        elif c_mode == 'color7':
            colormap = plt.cm.turbo
        elif c_mode == 'color8':
            colormap = plt.cm.nipy_spectral
        elif c_mode == 'color9':
            colormap = plt.cm.gist_ncar
        grid_color = 'silver'
        face_color = 'darkgrey'
        grid_alpha = 0.75
        if grid:
            grid_lines = '-.'
        else:
            grid_lines = 'None'

    # Create a dictionary to store the color for each track based on its confinement ratio
    track_colors = {}

    for track_ids_all in unique_tracks['TRACK_ID'].drop_duplicates():
        ratio = lut_norm_df[lut_norm_df['TRACK_ID'] == track_ids_all][lut_metric].values[0]
        track_colors[f'all all {track_ids_all}'] = colormap(norm(ratio))

    for condition in unique_tracks['CONDITION'].drop_duplicates():
        condition_tracks = unique_tracks[unique_tracks['CONDITION'] == condition]
        for track_ids_conditions in condition_tracks['TRACK_ID'].drop_duplicates():
            ratio = lut_norm_df[lut_norm_df['TRACK_ID'] == track_ids_conditions][lut_metric].values[0]
            track_colors[f'{condition} all {track_ids_conditions}'] = colormap(norm(ratio))
        
        for replicate in condition_tracks['REPLICATE'].drop_duplicates():
            replicate_tracks = condition_tracks[condition_tracks['REPLICATE'] == replicate]
            for track_id_replicates in replicate_tracks['TRACK_ID'].drop_duplicates():
                ratio = lut_norm_df[lut_norm_df['TRACK_ID'] == track_id_replicates][lut_metric].values[0]
                track_colors[f'{condition} {replicate} {track_id_replicates}'] = colormap(norm(ratio))

    # Set up the plot limits
    # Define the desired dimensions in microns
    microns_per_pixel = 0.7381885238402274 # for 10x lens
    x_min, x_max = 0, (1600 * microns_per_pixel)
    y_min, y_max = 0, (1200 * microns_per_pixel)

    ax.set_aspect('1', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Position X [microns]')
    ax.set_ylabel('Position Y [microns]')
    ax.set_title('Track Visualization', fontsize=title_size)
    ax.set_facecolor(face_color)
    ax.grid(True, which='both', axis='both', color=grid_color, linestyle=grid_lines, linewidth=1, alpha=grid_alpha)

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
    ax.tick_params(axis='both', which='major', labelsize=8)

    return fig, ax, unique_tracks, track_colors, norm, colormap

def visualize_tracks(df, df2, condition='all', replicate='all', c_mode='color1', grid=True, smoothing_index=0, lut_metric='NET_DISTANCE', lw=1, arrowsize=6):  # smoothened tracks visualization

    # try:
    #     condition = int(condition)
    #     replicate = int(replicate)
    # except ValueError or TypeError:
    #     pass

    if condition == None or replicate == None:
        pass
    else:
        try:
            condition = int(condition)
        except ValueError or TypeError:
            pass
        try:
            replicate = int(replicate)
        except ValueError or TypeError:
            pass

    if condition == 'all':
        df = df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate == 'all':
        df = df[df['CONDITION'] == condition].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])
    elif condition != 'all' and replicate != 'all':
        df = df[(df['CONDITION'] == condition) & (df['REPLICATE'] == replicate)].sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])


    # Using the  track_visuals function
    fig_visuals, ax_visuals, unique_tracks, track_colors_visuals, norm_visuals, colormap_visuals = track_visuals(df2, c_mode, grid, lut_metric)

    # Plot the full tracks
    for condition in df['CONDITION'].drop_duplicates():
        condition_tracks = df[df['CONDITION'] == condition]
        for replicate in condition_tracks['REPLICATE'].drop_duplicates():
            replicate_tracks = condition_tracks[condition_tracks['REPLICATE'] == replicate]
            for track_id in replicate_tracks['TRACK_ID'].drop_duplicates():
                track_data = replicate_tracks[replicate_tracks['TRACK_ID'] == track_id]
                x = track_data['POSITION_X']
                y = track_data['POSITION_Y']

                # Apply smoothing to the track (if applicable)
                if smoothing_index == None:
                    x_smoothed = x
                    y_smoothed = y
                elif smoothing_index > 0:
                    x_smoothed = x.rolling(window=smoothing_index, min_periods=1).mean()
                    y_smoothed = y.rolling(window=smoothing_index, min_periods=1).mean()
                else:
                    x_smoothed = x
                    y_smoothed = y

                ax_visuals.plot(x_smoothed, y_smoothed, lw=lw, color=track_colors_visuals[f'{condition} {replicate} {track_id}'], label=f'Track {track_id}')
                
                # Get the original color from track_colors_visuals[track_id]
                arrow_color = mcolors.to_rgb(track_colors_visuals[f'{condition} {replicate} {track_id}'])

                if len(x_smoothed) > 1:
                    # Extract the mean direction from df2 for the current track
                    mean_direction_rad = df2[df2[['CONDITION', 'REPLICATE', 'TRACK_ID']] == [condition, replicate, track_id]]['MEAN_DIRECTION_RAD'].values[0]
                    
                    # Use trigonometry to calculate the direction (dx, dy) from the angle
                    dx = np.cos(mean_direction_rad)  # Change in x based on angle
                    dy = np.sin(mean_direction_rad)  # Change in y based on angle
                    
                    # Create an arrow to indicate direction
                    arrow = FancyArrowPatch(
                        posA=(x_smoothed.iloc[-2], y_smoothed.iloc[-2]),  # Start position (second-to-last point)
                        posB=(x_smoothed.iloc[-2] + dx, y_smoothed.iloc[-2] + dy),  # End position based on direction
                        arrowstyle='-|>',  # Style of the arrow (you can adjust the style as needed)
                        color=arrow_color,  # Set the color of the arrow
                        mutation_scale=arrowsize,  # Scale the size of the arrow head (adjust this based on the plot scale)
                        linewidth=1.2,  # Line width for the arrow
                        zorder=30  # Ensure the arrow is drawn on top of the line
                    )

                    # Add the arrow to your plot (if you're using a `matplotlib` figure/axes)
                    plt.gca().add_patch(arrow)

    return plt.gcf()

def tracks_lut_map(df2, c_mode='color1', lut_metric='NET_DISTANCE', metrics_dict=None):

    lut_norm_df = df2[['TRACK_ID', lut_metric]].drop_duplicates()

    # Normalize the NET_DISTANCE to a 0-1 range
    lut_min = lut_norm_df[lut_metric].min()
    lut_max = lut_norm_df[lut_metric].max()
    norm = plt.Normalize(vmin=lut_min, vmax=lut_max)


    if c_mode == 'greyscale':
        colormap = plt.cm.gist_yarg
    else:
        if c_mode == 'color1':
            colormap = plt.cm.jet
        elif c_mode == 'color2':
            colormap = plt.cm.brg
        elif c_mode == 'color3':
            colormap = plt.cm.hot
        elif c_mode == 'color4':
            colormap = plt.cm.gnuplot
        elif c_mode == 'color5':
            colormap = plt.cm.viridis
        elif c_mode == 'color6':
            colormap = plt.cm.rainbow
        elif c_mode == 'color7':
            colormap = plt.cm.turbo
        elif c_mode == 'color8':
            colormap = plt.cm.nipy_spectral
        elif c_mode == 'color9':
            colormap = plt.cm.gist_ncar
    
    # Add a colorbar to show the LUT map
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    # Create a separate figure for the LUT map (colorbar)
    fig_lut, ax_lut = plt.subplots(figsize=(2, 6))
    ax_lut.axis('off')
    cbar = fig_lut.colorbar(sm, ax=ax_lut, orientation='vertical', extend='both')
    cbar.set_label(metrics_dict[lut_metric], fontsize=10)

    return plt.gcf()



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


# def swarm_plot(df, metric, Metric):
#     plt.figure(figsize=(12.5, 9.5))
    
#     # Ensure CONDITION is treated as categorical
#     df['CONDITION'] = df['CONDITION'].astype(str)

#     swarm_size = 3.15
#     swarm_alpha = 0.5

#     violin_fill_color = 'whitesmoke'
#     violin_edge_color = 'lightgrey'
#     violin_alpha = 0.525

#     mean_span = 0.275
#     median_span = 0.25
#     line_width = 1.6
    
#     sns.swarmplot(data=df, x='CONDITION', y=metric, hue='REPLICATE', palette='tab10', size=swarm_size, dodge=False, alpha=swarm_alpha, zorder=1, legend=False)
#     sns.despine()

#     replicate_means = df.groupby(['CONDITION', 'REPLICATE'])[metric].mean().reset_index()
#     sns.scatterplot(data=replicate_means, x='CONDITION', y=metric, hue='REPLICATE', palette='tab10', edgecolor='black', s=150, zorder=3, legend=False)
    
#     sns.violinplot(data=df, x='CONDITION', y=metric, color=violin_fill_color, edgecolor=violin_edge_color, inner=None, alpha=violin_alpha, zorder=0)

#     # Calculate mean and median for each condition
#     condition_stats = df.groupby('CONDITION')[metric].agg(['mean', 'median']).reset_index()

#     # Plot mean and median lines for each condition using seaborn functions
#     for i, row in condition_stats.iterrows():
#         x_center = i # Adjust x-coordinate to start at the correct condition position
#         sns.lineplot(x=[x_center - mean_span, x_center + mean_span], y=[row['mean'], row['mean']], color='black', linestyle='-', linewidth=line_width, zorder=4, label='Mean' if i == 0 else "")
#         sns.lineplot(x=[x_center - median_span, x_center + median_span], y=[row['median'], row['median']], color='black', linestyle='--', linewidth=line_width, zorder=4, label='Median' if i == 0 else "")

#     ''' P-test
#     # Perform pairwise p-tests
#     conditions = df['CONDITION'].unique()
#     pairs = list(combinations(conditions, 2))
#     y_max = df[metric].max()
#     y_offset = (y_max * 0.1)  # Offset for p-value annotations
#     for i, (cond1, cond2) in enumerate(pairs):
#         data1 = df[df['CONDITION'] == cond1][metric]
#         data2 = df[df['CONDITION'] == cond2][metric]
#         stat, p_value = mannwhitneyu(data1, data2)
        
#         # Annotate the plot with the p-value
#         x1, x2 = conditions.tolist().index(cond1), conditions.tolist().index(cond2)
#         y = y_max + y_offset * (i + 1)
#         plt.plot([x1, x1, x2, x2], [y+4.5, y + y_offset / 2.5, y + y_offset / 2.5, y+1.5], lw=1, color='black')
#         plt.text((x1 + x2) / 2, y + y_offset / 2, f'p = {round(p_value, 3):.3f}', ha='center', va='bottom', fontsize=10, color='black')
#     '''
    
#     plt.legend(loc='upper right')
#     plt.title(f"Swarm Plot with Mean and Median Lines for {Metric}")
#     plt.xlabel("Condition")
#     plt.ylabel(Metric)
#     return plt.gcf()


def swarm_plot(df, metric, Metric, show_violin=True, show_swarm=True, show_mean=True, show_median=True, show_error_bars=True, show_legend=True, p_testing=False):
    # fig, ax = plt.subplots(figsize=(12.5, 9.5))
    
    # Set the figure size
    plt.figure(figsize=(12.5, 9.5))
    
    # Ensure CONDITION is treated as categorical
    df['CONDITION'] = df['CONDITION'].astype(str)

    swarm_size = 3.15
    swarm_alpha = 0.5

    violin_fill_color = 'whitesmoke'
    violin_edge_color = 'lightgrey'
    violin_alpha = 0.525

    mean_span = 0.275
    median_span = 0.25
    line_width = 1.6

    sns.despine()

    if show_swarm:
        sns.swarmplot(data=df, x='CONDITION', y=metric, hue='REPLICATE', palette='tab10', size=swarm_size, dodge=False, alpha=swarm_alpha, zorder=1, legend=False)
    else:
        pass

    replicate_means = df.groupby(['CONDITION', 'REPLICATE'])[metric].mean().reset_index()
    sns.scatterplot(data=replicate_means, x='CONDITION', y=metric, hue='REPLICATE', palette='tab10', edgecolor='black', s=175, zorder=3, legend=False)

    if show_violin:
        sns.violinplot(data=df, x='CONDITION', y=metric, color=violin_fill_color, edgecolor=violin_edge_color, inner=None, alpha=violin_alpha, zorder=0)
    else:
        pass

    # Calculate mean and median for each condition
    condition_stats = df.groupby('CONDITION')[metric].agg(['mean', 'median']).reset_index()

    # Plot mean and median lines for each condition using seaborn functions
    for i, row in condition_stats.iterrows():
        x_center = i # Adjust x-coordinate to start at the correct condition position
        if show_mean:
            sns.lineplot(x=[x_center - mean_span, x_center + mean_span], y=[row['mean'], row['mean']], color='black', linestyle='-', linewidth=line_width, zorder=4, label='Mean' if i == 0 else "")
        else:
            pass
        if show_median:
            sns.lineplot(x=[x_center - median_span, x_center + median_span], y=[row['median'], row['median']], color='black', linestyle='--', linewidth=line_width, zorder=4, label='Median' if i == 0 else "")
        else:
            pass

    # Calculate mean, median, and standard deviation for each condition
    condition_stats = df.groupby('CONDITION')[metric].agg(['mean', 'median', 'std']).reset_index()

    if show_error_bars:
        # Plot error bars representing mean ± standard deviation for each condition
        for i, row in condition_stats.iterrows():
            x_center = i
            plt.errorbar(x_center, row['mean'], yerr=row['std'], fmt='None', color='black', alpha=0.8,
                        linewidth=1, capsize=11, zorder=5, label='Mean ± SD' if i == 0 else "")
    else:
        pass
    
    if p_testing:
        # P-test
        # Perform pairwise p-tests
        conditions = df['CONDITION'].unique()
        pairs = list(combinations(conditions, 2))
        y_max = df[metric].max()
        y_offset = (y_max * 0.1)  # Offset for p-value annotations
        for i, (cond1, cond2) in enumerate(pairs):
            data1 = df[df['CONDITION'] == cond1][metric]
            data2 = df[df['CONDITION'] == cond2][metric]
            stat, p_value = mannwhitneyu(data1, data2)
            
            # Annotate the plot with the p-value
            x1, x2 = conditions.tolist().index(cond1), conditions.tolist().index(cond2)
            y = y_max + y_offset * (i + 1)
            plt.plot([x1, x1, x2, x2], [y+4.5, y + y_offset / 2.5, y + y_offset / 2.5, y+1.5], lw=1, color='black')
            plt.text((x1 + x2) / 2, y + y_offset / 2, f'p = {round(p_value, 3):.3f}', ha='center', va='bottom', fontsize=10, color='black')
        else:
            pass
        

    plt.title(f"Swarm Plot with Mean and Median Lines for {Metric}")
    plt.xlabel("Condition")
    plt.ylabel(Metric)
    # plt.legend(title='Legend', title_fontsize='12', fontsize='10', loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)
    # sns.replot

    # Create a custom legend entry for the replicates marker.
    # Here, we choose the first color of the palette ('tab10') as representative.
    replicate_handle = mlines.Line2D([], [], marker='o', color='w',
                                     markerfacecolor=sns.color_palette('tab10')[0],
                                     markeredgecolor='black', markersize=10,
                                     label='Replicates')

    # Get current handles and labels (from Mean, Median, and Error Bars)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the custom replicates handle
    handles.insert(0, replicate_handle)
    labels.insert(0, 'Replicates')
    
    plt.legend(handles=handles, labels=labels,
               title='Legend', title_fontsize='12', fontsize='10',
               loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)
    
    if show_legend:
        pass
    else:
        plt.legend().remove()

    return plt.gcf()

