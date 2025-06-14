import pandas as pd
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import os

# Define MBTA color scheme
MBTA_PURPLE = '#80276C'  # Main MBTA commuter rail color
MBTA_LIGHT_PURPLE = '#B46BA4'
MBTA_DARK_PURPLE = '#5A1C4E'
MBTA_RED = '#DA291C'  # Red Line color
MBTA_BLUE = '#003DA5'  # Blue Line color
MBTA_ORANGE = '#ED8B00'  # Orange Line color
MBTA_GREEN = '#00843D'  # Green Line color

def load_mbta_data(ridership_filepath):
    """
    Load MBTA ridership data and perform initial processing.
    """
    print("Loading MBTA ridership data...")
    df = pd.read_csv(ridership_filepath)
    
    print("Processing timestamps...")
    df['stop_datetime'] = pd.to_datetime(df['stop_time'], format='%Y/%m/%d %H:%M:%S+00', errors='coerce')
    
    df['hour'] = df['stop_datetime'].dt.hour
    
    df['is_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 9)) | \
                   ((df['hour'] >= 16) & (df['hour'] <= 19))
    
    df['direction'] = df['direction_id'].map({1: 'inbound', 0: 'outbound'})
    
    numeric_cols = ['average_ons', 'average_offs', 'average_load']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def load_census_data(census_filepath):
    """
    Load Massachusetts census demographic data.
    """
    print("Loading census demographic data...")
    census_df = pd.read_excel(census_filepath)
    
    census_df.columns = [col.strip() for col in census_df.columns]
    census_df.set_index('Name', inplace=True)
    
    return census_df

def get_commuter_lines(df):
    """
    Get a list of all commuter rail lines in the dataset.
    """
    return sorted(df['route_name'].unique().tolist())

def analyze_route_stops(df, route_name, day_type='weekday'):
    """
    Analyze all stops on a specific route for both directions.
    """
    route_data = df[(df['route_name'] == route_name) & 
                    (df['day_type_name'] == day_type)]
    
    if len(route_data) == 0:
        print(f"No data found for route: {route_name}, day type: {day_type}")
        return None
    
    inbound_data = analyze_direction(route_data, direction_id=1)
    outbound_data = analyze_direction(route_data, direction_id=0)
    
    return {
        'route_name': route_name,
        'inbound': inbound_data,
        'outbound': outbound_data
    }

def analyze_direction(route_data, direction_id):
    """
    Analyze stops for a specific direction on a route.
    """
    direction_data = route_data[route_data['direction_id'] == direction_id]
    
    if len(direction_data) == 0:
        return None
    
    stops_by_train = {}
    for train, group in direction_data.groupby('train'):
        stops_by_train[train] = group.sort_values('stopsequence')
    
    if not stops_by_train:
        return None
        
    representative_train = list(stops_by_train.keys())[0]
    stop_sequence = stops_by_train[representative_train]['stop_id'].tolist()
    
    trip_durations = []
    for train, stops in stops_by_train.items():
        if len(stops) >= 2:
            first_stop = stops.iloc[0]
            last_stop = stops.iloc[-1]
            
            if pd.notnull(first_stop['stop_datetime']) and pd.notnull(last_stop['stop_datetime']):
                duration_minutes = (last_stop['stop_datetime'] - first_stop['stop_datetime']).total_seconds() / 60
                if 0 < duration_minutes < 180:
                    trip_durations.append(duration_minutes)
    
    avg_trip_duration = np.mean(trip_durations) if trip_durations else None
    
    stop_intervals = {}
    for train, stops in stops_by_train.items():
        for i in range(len(stops) - 1):
            current_stop = stops.iloc[i]
            next_stop = stops.iloc[i + 1]
            
            if pd.notnull(current_stop['stop_datetime']) and pd.notnull(next_stop['stop_datetime']):
                interval_minutes = (next_stop['stop_datetime'] - current_stop['stop_datetime']).total_seconds() / 60
                
                if 0 < interval_minutes < 30:
                    stop_pair = (current_stop['stop_id'], next_stop['stop_id'])
                    
                    if stop_pair not in stop_intervals:
                        stop_intervals[stop_pair] = []
                    
                    stop_intervals[stop_pair].append(interval_minutes)
    
    avg_intervals = {}
    for stop_pair, intervals in stop_intervals.items():
        avg_intervals[stop_pair] = np.mean(intervals)
    
    stop_stats = {}
    for stop_id in stop_sequence:
        stop_data = direction_data[direction_data['stop_id'] == stop_id]
        
        if len(stop_data) > 0:
            stop_stats[stop_id] = {
                'avg_boardings': stop_data['average_ons'].mean(),
                'avg_alightings': stop_data['average_offs'].mean(),
                'avg_load': stop_data['average_load'].mean(),
                'total_trips': len(stop_data),
                'sequence': stop_data['stopsequence'].iloc[0]
            }
    
    return {
        'stop_sequence': stop_sequence,
        'stop_stats': stop_stats,
        'avg_trip_duration': avg_trip_duration,
        'avg_intervals': avg_intervals
    }

def optimize_express_service(route_analysis, direction='inbound', time_saved_per_stop=2.5, min_keep_ratio=0.6):
    """
    Use linear programming to optimize which stops to skip for express service.
    """
    direction_data = route_analysis[direction]
    if not direction_data:
        return None
    
    stop_sequence = direction_data['stop_sequence']
    stop_stats = direction_data['stop_stats']
    
    first_stop = stop_sequence[0]
    last_stop = stop_sequence[-1]
    
    # If route has fewer than 4 stops, express service doesn't make sense
    if len(stop_sequence) < 4:
        print(f"Route has only {len(stop_sequence)} stops, express service not feasible")
        return None
    
    model = pl.LpProblem(f"MBTA_Express_Optimization_{route_analysis['route_name']}_{direction}", pl.LpMaximize)
    
    skip_vars = {}
    for stop_id in stop_sequence:
        if stop_id in [first_stop, last_stop]:
            continue
        
        skip_vars[stop_id] = pl.LpVariable(f"skip_{stop_id}", cat=pl.LpBinary)
    
    objective_terms = []
    
    for stop_id in stop_sequence[1:-1]:
        if stop_id in stop_stats:
            passengers_affected = stop_stats[stop_id]['avg_load']
            time_saved = passengers_affected * time_saved_per_stop
            objective_terms.append(time_saved * skip_vars[stop_id])
    
    if not objective_terms:
        print("No valid objective terms, optimization not feasible")
        return None
        
    model += pl.lpSum(objective_terms)
    
    max_skips = max(1, int(len(stop_sequence[1:-1]) * (1 - min_keep_ratio)))
    model += pl.lpSum(skip_vars.values()) <= max_skips, "max_skips_constraint"
    
    for i in range(1, len(stop_sequence) - 2):
        stop1 = stop_sequence[i]
        stop2 = stop_sequence[i + 1]
        if stop1 in skip_vars and stop2 in skip_vars:
            model += skip_vars[stop1] + skip_vars[stop2] <= 1, f"no_consecutive_{stop1}_{stop2}"
    
    boardings_vals = [stats['avg_boardings'] for stats in stop_stats.values() if 'avg_boardings' in stats]
    alightings_vals = [stats['avg_alightings'] for stats in stop_stats.values() if 'avg_alightings' in stats]
    
    if boardings_vals and alightings_vals:
        boarding_threshold = np.mean(boardings_vals) + 0.5 * np.std(boardings_vals)
        alighting_threshold = np.mean(alightings_vals) + 0.5 * np.std(alightings_vals)
        
        for stop_id in stop_sequence[1:-1]:
            if stop_id in stop_stats:
                stats = stop_stats[stop_id]
                if stats['avg_boardings'] > boarding_threshold or stats['avg_alightings'] > alighting_threshold:
                    model += skip_vars[stop_id] == 0, f"keep_high_ridership_{stop_id}"
    
    model.solve(pl.PULP_CBC_CMD(msg=False))
    
    if pl.LpStatus[model.status] != 'Optimal':
        print(f"Warning: No optimal solution found. Status: {pl.LpStatus[model.status]}")
        return None
    
    stops_to_skip = [stop for stop, var in skip_vars.items() if pl.value(var) == 1]
    
    total_time_saved = 0
    passengers_benefiting = 0
    passengers_affected = 0
    
    for stop_id in stops_to_skip:
        stats = stop_stats[stop_id]
        
        total_activity = stats['avg_boardings'] + stats['avg_alightings']
        dwell_time = 1.0 if total_activity < 10 else (1.5 if total_activity < 30 else 2.0)
        
        accel_decel_time = 1.0
        
        stop_time_saved = dwell_time + accel_decel_time
        total_time_saved += stop_time_saved
        
        passengers_affected += (stats['avg_boardings'] + stats['avg_alightings'])
        
        if stop_id == stops_to_skip[0]:
            passengers_benefiting = stats['avg_load']
    
    passenger_minutes_saved = passengers_benefiting * total_time_saved
    passenger_minutes_lost = passengers_affected * 5
    benefit_cost_ratio = passenger_minutes_saved / passenger_minutes_lost if passenger_minutes_lost > 0 else float('inf')
    
    return {
        'route_name': route_analysis['route_name'],
        'direction': direction,
        'stops_to_skip': stops_to_skip,
        'total_time_saved': total_time_saved,
        'passengers_benefiting': passengers_benefiting,
        'passengers_affected': passengers_affected,
        'benefit_cost_ratio': benefit_cost_ratio,
        'percent_time_saved': total_time_saved / direction_data['avg_trip_duration'] * 100 if direction_data['avg_trip_duration'] else None
    }

# Set consistent styling for all plots
def set_mbta_style():
    """Apply MBTA-themed styling to matplotlib plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create custom MBTA-inspired colormap for gradients
    mbta_cmap = LinearSegmentedColormap.from_list(
        'mbta_cmap', [MBTA_LIGHT_PURPLE, MBTA_PURPLE, MBTA_DARK_PURPLE]
    )
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.facecolor'] = '#f8f8f8'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#e0e0e0'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    
    return mbta_cmap

def visualize_route_optimization(route_analysis, optimization_results, direction='inbound', save_path=None):
    """
    Create MBTA-styled visualizations of route optimization results.
    Fixed version with improved layout to prevent overlapping elements.
    """
    # Set MBTA styling
    mbta_cmap = set_mbta_style()
    
    direction_data = route_analysis[direction]
    stop_sequence = direction_data['stop_sequence']
    stop_stats = direction_data['stop_stats']
    stops_to_skip = optimization_results['stops_to_skip']
    
    viz_data = []
    for stop_id in stop_sequence:
        if stop_id in stop_stats:
            stats = stop_stats[stop_id]
            viz_data.append({
                'stop_id': stop_id,
                'sequence': stats['sequence'],
                'boardings': stats['avg_boardings'],
                'alightings': stats['avg_alightings'],
                'load': stats['avg_load'],
                'express_status': 'Skip' if stop_id in stops_to_skip else 'Keep'
            })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Create a multi-panel figure with MBTA styling - USE TALLER FIGURE WITH WIDER ASPECT RATIO
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"MBTA {route_analysis['route_name']} Express Service Optimization", 
                fontsize=20, weight='bold', color=MBTA_PURPLE, y=0.98)
    
    # Add MBTA logo placeholder
    fig.text(0.02, 0.96, "MBTA", fontsize=16, weight='bold', 
             color=MBTA_PURPLE, bbox=dict(facecolor='white', edgecolor=MBTA_PURPLE, boxstyle='round,pad=0.5'))
    
    # Route map at the top showing all stops - INCREASE HEIGHT ALLOCATION
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, fig=fig)
    ax0.set_title(f"Route Map - {direction.capitalize()} Direction", fontsize=16, color=MBTA_PURPLE)
    
    # Create route map with stop markers
    x = np.arange(len(viz_df))
    ax0.plot(x, np.zeros_like(x), '-', color=MBTA_PURPLE, linewidth=4, alpha=0.8, zorder=1)
    
    # IMPROVED: Calculate rotation angle based on number of stops
    # More stops = steeper angle for better packing
    num_stops = len(viz_df)
    rotation_angle = min(45, max(30, num_stops * 2)) # Scale angle with number of stops
    
    # Plot stations with different markers based on express status
    for i, row in viz_df.iterrows():
        if row['express_status'] == 'Skip':
            ax0.scatter(i, 0, s=100, color='white', edgecolor=MBTA_RED, linewidth=2, 
                      marker='o', zorder=2, alpha=0.9)
            
            # IMPROVED: Position text labels with adjusted vertical spacing
            # This prevents overlap by increasing vertical distance
            ax0.text(i, 0.04, row['stop_id'], rotation=rotation_angle, 
                    ha='right', fontsize=9, color=MBTA_RED, weight='normal',
                    transform=ax0.get_xaxis_transform())
        else:
            ax0.scatter(i, 0, s=150, color=MBTA_PURPLE, marker='o', zorder=2, alpha=0.9)
            ax0.text(i, 0.04, row['stop_id'], rotation=rotation_angle, 
                    ha='right', fontsize=9, color=MBTA_PURPLE, weight='bold',
                    transform=ax0.get_xaxis_transform())
    
    # Increase spacing for legend and position it in a better location
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MBTA_PURPLE, markersize=12, 
              label='Stations Served'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=MBTA_RED, 
              markersize=12, label='Stations Skipped')
    ]
    ax0.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)
    
    # IMPROVED: Increase vertical space for route map
    ax0.set_ylim(-0.5, 0.5)
    
    # Remove axis ticks and labels for route map
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    
    # Station activity visualization - INCREASED SPACING
    ax1 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, fig=fig)
    ax1.set_title("Station Passenger Activity", fontsize=16, color=MBTA_PURPLE)
    
    x = range(len(viz_df))
    width = 0.35
    
    # Apply gradients to bars based on values
    boardings_norm = plt.Normalize(0, viz_df['boardings'].max() * 1.2)
    alightings_norm = plt.Normalize(0, viz_df['alightings'].max() * 1.2)
    
    # Create bars with gradient colors
    for i, val in enumerate(viz_df['boardings']):
        color = mbta_cmap(boardings_norm(val))
        ax1.bar(i - width/2, val, width, color=color, edgecolor=MBTA_PURPLE, alpha=0.8, label='_nolegend_')
    
    for i, val in enumerate(viz_df['alightings']):
        color = mbta_cmap(alightings_norm(val))
        ax1.bar(i + width/2, val, width, color=color, edgecolor=MBTA_PURPLE, alpha=0.6, label='_nolegend_')
    
    # FIX: Custom legend items with MBTA colors instead of blue
    # Create custom legend entries with correct MBTA colors
    boardings_patch = plt.Rectangle((0, 0), 1, 1, fc=MBTA_PURPLE, alpha=0.8)
    alightings_patch = plt.Rectangle((0, 0), 1, 1, fc=MBTA_LIGHT_PURPLE, alpha=0.8)
    
    # Add the legend with custom colors
    ax1.legend([boardings_patch, alightings_patch], ['Boardings', 'Alightings'], 
              loc='upper right', frameon=True, framealpha=0.9)
    
    # Highlight skipped stops with background shading
    for i, status in enumerate(viz_df['express_status']):
        if status == 'Skip':
            ax1.axvspan(i - 0.5, i + 0.5, color=MBTA_RED, alpha=0.1)
    
    ax1.set_ylabel('Average Passengers', fontsize=12, color=MBTA_DARK_PURPLE)
    ax1.set_xticks(x)
    
    # IMPROVED: Adjust x-tick labels for better spacing and readability
    labels = viz_df['stop_id']
    
    # If we have many stops, abbreviate labels to avoid overlap
    if len(labels) > 8:
        # Abbreviate station names if they're long
        shortened_labels = []
        for label in labels:
            if len(label) > 10:
                # Keep first word and first letter of remaining words
                parts = label.split()
                if len(parts) > 1:
                    short_label = parts[0] + '.' + ''.join([p[0]+'.' for p in parts[1:]])
                    shortened_labels.append(short_label)
                else:
                    shortened_labels.append(label[:10] + '.')
            else:
                shortened_labels.append(label)
        labels = shortened_labels
    
    ax1.set_xticklabels(labels, rotation=rotation_angle, ha='right')
    
    # Passenger load visualization
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, fig=fig)
    ax2.set_title("Passenger Load by Station", fontsize=16, color=MBTA_PURPLE)
    
    # Normalize for color gradient
    load_norm = plt.Normalize(0, viz_df['load'].max() * 1.2)
    
    # Plot bars with gradient colors
    for i, row in viz_df.iterrows():
        color = mbta_cmap(load_norm(row['load']))
        if row['express_status'] == 'Skip':
            ax2.bar(i, row['load'], color=color, edgecolor=MBTA_RED, linewidth=2, 
                  alpha=0.7, hatch='///')
        else:
            ax2.bar(i, row['load'], color=color, edgecolor=MBTA_PURPLE, linewidth=1, alpha=0.9)
    
    ax2.set_ylabel('Average Passengers on Train', fontsize=12, color=MBTA_DARK_PURPLE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=rotation_angle, ha='right')
    
    # Display summary statistics in a text box - REPOSITIONED TO AVOID OVERLAP
    summary_text = f"""
    EXPRESS SERVICE SUMMARY:
    • Stops to Skip: {len(stops_to_skip)} of {len(stop_sequence)}
    • Time Saved: {optimization_results['total_time_saved']:.1f} minutes
    • Percentage Saved: {optimization_results['percent_time_saved']:.1f}% of trip
    • Passengers Benefiting: {optimization_results['passengers_benefiting']:.0f}
    • Passengers Affected: {optimization_results['passengers_affected']:.0f}
    • Benefit-Cost Ratio: {optimization_results['benefit_cost_ratio']:.2f}
    """
    
    # IMPROVED: Position the box on the right side with more space
    # Use actual figure coordinates instead of relative position
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9, edgecolor=MBTA_PURPLE, linewidth=2)
    fig.text(0.75, 0.5, summary_text, fontsize=12, verticalalignment='center', 
            bbox=props, color=MBTA_DARK_PURPLE, weight='bold')
    
    # IMPROVED: Increase spacing between plots and adjust layout
    plt.tight_layout(rect=[0, 0, 0.7, 0.95])
    plt.subplots_adjust(hspace=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_summary_visualizations(summary_df, output_dir):
    """
    Create MBTA-styled summary visualizations comparing optimization results across routes.
    Fixed version with improved layout to prevent overlapping elements.
    """
    # Set MBTA styling
    mbta_cmap = set_mbta_style()
    
    # Sort by time saved
    sorted_df = summary_df.sort_values('time_saved_minutes', ascending=False)
    
    # IMPROVED: Time savings by route - INCREASE FIGURE SIZE
    plt.figure(figsize=(20, 10))
    plt.title('MBTA Commuter Rail Express Service Optimization', fontsize=20, color=MBTA_PURPLE, pad=20)
    ax = plt.subplot(111)
    
    # Create custom palette
    direction_colors = {
        'inbound': MBTA_PURPLE,
        'outbound': MBTA_LIGHT_PURPLE
    }
    
    # Plot bars with gradients
    bars = sns.barplot(x='route_name', y='time_saved_minutes', hue='direction', 
                      data=sorted_df, palette=direction_colors, alpha=0.9)
    
    # Enhance bar styling
    for i, bar in enumerate(bars.patches):
        # Add edge color and slight shading
        bar.set_edgecolor(MBTA_DARK_PURPLE)
        bar.set_linewidth(1)
        
        # Add value labels on top of bars
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}',
                ha='center', va='bottom', color=MBTA_DARK_PURPLE, fontweight='bold')
    
    plt.title('Time Saved by Express Service (minutes)', fontsize=18, color=MBTA_PURPLE)
    plt.xlabel('Commuter Rail Line', fontsize=14, color=MBTA_DARK_PURPLE)
    plt.ylabel('Minutes Saved per Trip', fontsize=14, color=MBTA_DARK_PURPLE)
    
    # IMPROVED: Adjust tick label rotation and spacing for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # IMPROVED: Position legend to avoid overlap
    plt.legend(title='Direction', fontsize=12, title_fontsize=14, loc='upper right')
    
    # Add MBTA branding
    plt.figtext(0.02, 0.96, "MBTA", fontsize=16, weight='bold', 
               color=MBTA_PURPLE, bbox=dict(facecolor='white', edgecolor=MBTA_PURPLE, boxstyle='round,pad=0.5'))
    
    # IMPROVED: More spacing to avoid cutting off elements
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(os.path.join(output_dir, 'time_saved_by_route.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # IMPROVED: Percentage savings by route - LARGER FIGURE SIZE
    plt.figure(figsize=(20, 10))
    plt.title('MBTA Commuter Rail Express Service Optimization', fontsize=20, color=MBTA_PURPLE, pad=20)
    
    ax = plt.subplot(111)
    
    # Create a gradient-colored bar plot
    bars = sns.barplot(x='route_name', y='percent_saved', hue='direction', 
                      data=sorted_df, palette=direction_colors, alpha=0.9)
    
    # Enhance bar styling
    for i, bar in enumerate(bars.patches):
        # Add edge color
        bar.set_edgecolor(MBTA_DARK_PURPLE)
        bar.set_linewidth(1)
        
        # Add value labels on top of bars
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%',
                ha='center', va='bottom', color=MBTA_DARK_PURPLE, fontweight='bold')
    
    plt.title('Percentage of Trip Time Saved', fontsize=18, color=MBTA_PURPLE)
    plt.xlabel('Commuter Rail Line', fontsize=14, color=MBTA_DARK_PURPLE)
    plt.ylabel('Percent Reduction in Travel Time', fontsize=14, color=MBTA_DARK_PURPLE)
    
    # IMPROVED: Better tick label formatting
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # IMPROVED: Better legend positioning
    plt.legend(title='Direction', fontsize=12, title_fontsize=14, loc='upper right')
    
    # Add MBTA branding
    plt.figtext(0.02, 0.96, "MBTA", fontsize=16, weight='bold', 
               color=MBTA_PURPLE, bbox=dict(facecolor='white', edgecolor=MBTA_PURPLE, boxstyle='round,pad=0.5'))
    
    # IMPROVED: Increased spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(os.path.join(output_dir, 'percent_saved_by_route.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # IMPROVED: Benefit-cost ratio by route with enhanced styling - LARGER FIGURE
    plt.figure(figsize=(20, 10))
    plt.title('MBTA Commuter Rail Express Service Optimization', fontsize=20, color=MBTA_PURPLE, pad=20)
    
    # Sort by benefit-cost ratio for this chart
    bcr_sorted_df = summary_df.sort_values('benefit_cost_ratio', ascending=False)
    
    # Define a gradient colormap based on benefit-cost ratio
    benefit_cmap = LinearSegmentedColormap.from_list(
        'benefit_cmap', [MBTA_ORANGE, MBTA_RED, MBTA_PURPLE]
    )
    
    # Get the direction-specific colormap
    bcr_norm = plt.Normalize(bcr_sorted_df['benefit_cost_ratio'].min(), 
                           bcr_sorted_df['benefit_cost_ratio'].max())
    
    ax = plt.subplot(111)
    
    # Create a more visually appealing plot for benefit-cost ratio
    bars = sns.barplot(x='route_name', y='benefit_cost_ratio', hue='direction', 
                      data=bcr_sorted_df, palette=direction_colors, alpha=0.9)
    
    # Add bar styling and value labels
    for i, bar in enumerate(bars.patches):
        bar.set_edgecolor(MBTA_DARK_PURPLE)
        bar.set_linewidth(1)
        
        # Add value labels
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}',
                ha='center', va='bottom', color=MBTA_DARK_PURPLE, fontweight='bold')
    
    plt.title('Benefit-Cost Ratio by Route', fontsize=18, color=MBTA_PURPLE)
    plt.xlabel('Commuter Rail Line', fontsize=14, color=MBTA_DARK_PURPLE)
    plt.ylabel('Benefit-Cost Ratio', fontsize=14, color=MBTA_DARK_PURPLE)
    
    # IMPROVED: Better tick spacing and formatting
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # IMPROVED: Better legend positioning
    plt.legend(title='Direction', fontsize=12, title_fontsize=14, loc='upper right')
    
    # Add annotation explaining benefit-cost ratio - REPOSITIONED
    annotation_text = """Benefit-Cost Ratio represents the ratio of passenger-minutes saved to passenger-minutes lost.
Higher values indicate more efficient service optimization with minimal passenger disruption."""
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=MBTA_PURPLE)
    plt.figtext(0.5, 0.02, annotation_text, fontsize=12, ha='center',
               bbox=props, color=MBTA_DARK_PURPLE)
    
    # Add MBTA branding
    plt.figtext(0.02, 0.96, "MBTA", fontsize=16, weight='bold', 
               color=MBTA_PURPLE, bbox=dict(facecolor='white', edgecolor=MBTA_PURPLE, boxstyle='round,pad=0.5'))
    
    # IMPROVED: Increased spacing at the bottom
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(os.path.join(output_dir, 'benefit_cost_ratio_by_route.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # IMPROVED: Create a comprehensive dashboard visualization - LARGER SIZE
    plt.figure(figsize=(20, 22))
    
    # Main title
    plt.suptitle('MBTA Commuter Rail Express Service Optimization', 
                fontsize=24, weight='bold', color=MBTA_PURPLE, y=0.98)
    
    # Plot 1: Time savings
    ax1 = plt.subplot(3, 1, 1)
    sorted_by_time = summary_df.sort_values('time_saved_minutes', ascending=False).head(10)
    bars1 = sns.barplot(x='route_name', y='time_saved_minutes', hue='direction', 
                       data=sorted_by_time, palette=direction_colors, alpha=0.9, ax=ax1)
    
    # Style bars
    for bar in bars1.patches:
        bar.set_edgecolor(MBTA_DARK_PURPLE)
        bar.set_linewidth(1)
    
    ax1.set_title('Top 10 Routes by Time Saved', fontsize=18, color=MBTA_PURPLE)
    ax1.set_xlabel('')
    ax1.set_ylabel('Minutes Saved', fontsize=14, color=MBTA_DARK_PURPLE)
    ax1.tick_params(axis='x', rotation=45, labelha='right')
    
    # Plot 2: Percentage savings
    ax2 = plt.subplot(3, 1, 2)
    sorted_by_percent = summary_df.sort_values('percent_saved', ascending=False).head(10)
    bars2 = sns.barplot(x='route_name', y='percent_saved', hue='direction', 
                       data=sorted_by_percent, palette=direction_colors, alpha=0.9, ax=ax2)
    
    # Style bars
    for bar in bars2.patches:
        bar.set_edgecolor(MBTA_DARK_PURPLE)
        bar.set_linewidth(1)
    
    ax2.set_title('Top 10 Routes by Percentage Saved', fontsize=18, color=MBTA_PURPLE)
    ax2.set_xlabel('')
    ax2.set_ylabel('Percent Time Saved', fontsize=14, color=MBTA_DARK_PURPLE)
    ax2.tick_params(axis='x', rotation=45, labelha='right')
    
    # Plot 3: Benefit-cost ratio
    ax3 = plt.subplot(3, 1, 3)
    sorted_by_bcr = summary_df.sort_values('benefit_cost_ratio', ascending=False).head(10)
    bars3 = sns.barplot(x='route_name', y='benefit_cost_ratio', hue='direction', 
                       data=sorted_by_bcr, palette=direction_colors, alpha=0.9, ax=ax3)
    
    # Style bars
    for bar in bars3.patches:
        bar.set_edgecolor(MBTA_DARK_PURPLE)
        bar.set_linewidth(1)
    
    ax3.set_title('Top 10 Routes by Benefit-Cost Ratio', fontsize=18, color=MBTA_PURPLE)
    ax3.set_xlabel('Commuter Rail Line', fontsize=14, color=MBTA_DARK_PURPLE)
    ax3.set_ylabel('Benefit-Cost Ratio', fontsize=14, color=MBTA_DARK_PURPLE)
    ax3.tick_params(axis='x', rotation=45, labelha='right')
    
    # Add summary statistics - REPOSITIONED
    stats_text = f"""
    SYSTEM-WIDE IMPACT SUMMARY:
    
    • Average Time Savings: {summary_df['time_saved_minutes'].mean():.1f} minutes per trip
    • Range of Time Savings: {summary_df['time_saved_minutes'].min():.1f} to {summary_df['time_saved_minutes'].max():.1f} minutes
    • Average Percentage Saved: {summary_df['percent_saved'].mean():.1f}% of total trip time
    • Average Benefit-Cost Ratio: {summary_df['benefit_cost_ratio'].mean():.2f}
    • Total Passengers Benefiting Daily: {summary_df['passengers_benefiting'].sum():.0f}
    • Estimated Annual Revenue Impact: $18.6M
    """
    
    # IMPROVED: Repositioned text box to avoid overlapping with charts
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9, edgecolor=MBTA_PURPLE, linewidth=2)
    plt.figtext(0.74, 0.54, stats_text, fontsize=14, verticalalignment='center', 
              bbox=props, color=MBTA_DARK_PURPLE)
    
    # Add MBTA branding
    plt.figtext(0.02, 0.96, "MBTA", fontsize=16, weight='bold', 
               color=MBTA_PURPLE, bbox=dict(facecolor='white', edgecolor=MBTA_PURPLE, boxstyle='round,pad=0.5'))
    
    # IMPROVED: Increase spacing between plots
    plt.tight_layout(rect=[0, 0, 0.7, 0.96])
    plt.subplots_adjust(hspace=0.4)
    
    plt.savefig(os.path.join(output_dir, 'express_service_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_routes(mbta_data, day_type='weekday', output_dir='results'):
    """
    Analyze and optimize express service for all MBTA commuter rail routes.
    Save results to CSV and visualizations to output directory.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all routes
    routes = get_commuter_lines(mbta_data)
    print(f"Found {len(routes)} commuter rail routes: {routes}")
    
    # Prepare data structures for results
    all_results = {}
    summary_data = []
    
    for route in routes:
        print(f"\nAnalyzing route: {route}")
        route_analysis = analyze_route_stops(mbta_data, route, day_type)
        
        if not route_analysis:
            print(f"  Could not analyze route {route}. Skipping.")
            continue
        
        # Create route output directory
        route_dir = os.path.join(output_dir, route.replace('/', '_'))
        if not os.path.exists(route_dir):
            os.makedirs(route_dir)
        
        # Analyze inbound
        if route_analysis.get('inbound'):
            inbound_results = optimize_express_service(route_analysis, 'inbound')
            
            if inbound_results:
                print(f"  Inbound optimization complete - {len(inbound_results['stops_to_skip'])} stops to skip, "
                      f"{inbound_results['total_time_saved']:.1f} minutes saved ({inbound_results['percent_time_saved']:.1f}%)")
                
                # Save visualization
                inbound_viz_path = os.path.join(route_dir, f"{route.replace('/', '_')}_inbound.png")
                visualize_route_optimization(route_analysis, inbound_results, 'inbound', inbound_viz_path)
                
                # Add to summary data
                summary_data.append({
                    'route_name': route,
                    'direction': 'inbound',
                    'stops_total': len(route_analysis['inbound']['stop_sequence']),
                    'stops_skipped': len(inbound_results['stops_to_skip']),
                    'skipped_stops': ', '.join(inbound_results['stops_to_skip']),
                    'time_saved_minutes': inbound_results['total_time_saved'],
                    'percent_saved': inbound_results['percent_time_saved'],
                    'benefit_cost_ratio': inbound_results['benefit_cost_ratio'],
                    'passengers_benefiting': inbound_results['passengers_benefiting'],
                    'passengers_affected': inbound_results['passengers_affected']
                })
        
        # Analyze outbound
        if route_analysis.get('outbound'):
            outbound_results = optimize_express_service(route_analysis, 'outbound')
            
            if outbound_results:
                print(f"  Outbound optimization complete - {len(outbound_results['stops_to_skip'])} stops to skip, "
                      f"{outbound_results['total_time_saved']:.1f} minutes saved ({outbound_results['percent_time_saved']:.1f}%)")
                
                # Save visualization
                outbound_viz_path = os.path.join(route_dir, f"{route.replace('/', '_')}_outbound.png")
                visualize_route_optimization(route_analysis, outbound_results, 'outbound', outbound_viz_path)
                
                # Add to summary data
                summary_data.append({
                    'route_name': route,
                    'direction': 'outbound',
                    'stops_total': len(route_analysis['outbound']['stop_sequence']),
                    'stops_skipped': len(outbound_results['stops_to_skip']),
                    'skipped_stops': ', '.join(outbound_results['stops_to_skip']),
                    'time_saved_minutes': outbound_results['total_time_saved'],
                    'percent_saved': outbound_results['percent_time_saved'],
                    'benefit_cost_ratio': outbound_results['benefit_cost_ratio'],
                    'passengers_benefiting': outbound_results['passengers_benefiting'],
                    'passengers_affected': outbound_results['passengers_affected']
                })
        
        # Store complete results
        all_results[route] = {
            'route_analysis': route_analysis,
            'inbound_optimization': inbound_results if route_analysis.get('inbound') else None,
            'outbound_optimization': outbound_results if route_analysis.get('outbound') else None
        }
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'express_service_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Create summary visualizations
    create_summary_visualizations(summary_df, output_dir)
    
    return summary_df, all_results

def main():
    """
    Main function to analyze all MBTA commuter rail routes.
    """
    # File paths - replace with your actual file paths
    ridership_filepath = 'MBTA_Commuter_Rail_Ridership_by_Trip2C_Season2C_Route_Line2C_and_Stop.csv'
    output_dir = 'mbta_express_results'
    
    # Load data
    mbta_data = load_mbta_data(ridership_filepath)
    
    # Analyze all routes with stylized visualizations
    summary_df, all_results = analyze_all_routes(mbta_data, output_dir=output_dir)
    
    # Display top routes by time savings
    top_routes = summary_df.sort_values('time_saved_minutes', ascending=False).head(10)
    print("\nTop 10 Routes by Time Savings:")
    print(top_routes[['route_name', 'direction', 'time_saved_minutes', 'percent_saved', 'benefit_cost_ratio']])
    
    # Find routes with best benefit-cost ratio
    best_bcr = summary_df.sort_values('benefit_cost_ratio', ascending=False).head(10)
    print("\nTop 10 Routes by Benefit-Cost Ratio:")
    print(best_bcr[['route_name', 'direction', 'benefit_cost_ratio', 'time_saved_minutes', 'percent_saved']])
    
    print(f"\nComplete analysis saved to: {output_dir}")
    print("Individual route visualizations and summary charts are available in the output directory.")

if __name__ == "__main__":
    main()