"""
Plotly template for visualizing multiple time series data with a shared X-axis.
Includes unified hover and spikelines to easily compare data across different plots
at the exact same timestamp.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

def plot_synced_timeseries(
    df: pd.DataFrame,
    x_col: str,
    subplot_configs: List[Dict[str, Any]],
    title: str = "Shared X-Axis Time Series Analysis",
    height: int = 600,
    width: int = None,
    hovermode: str = "x unified",
) -> go.Figure:
    """
    Create a Plotly figure with multiple vertically-stacked subplots sharing an X-axis.
    A unified hover and vertical spikeline span across all plots to compare timestamps easily.
    
    Args:
        df: The pandas DataFrame containing the data.
        x_col: The name of the column containing the datetime/timestamp for the X-axis.
        subplot_configs: A list of dictionaries, where each dictionary represents a subplot.
            Each dictionary can have:
            - 'traces': List of kwargs for go.Scatter (e.g., [{'name': 'Speed', 'y': 'speed'}])
            - 'title': (Optional) String title for the subplot
            - 'y_label': (Optional) String label for the Y-axis
        title: Main title of the entire figure.
        height: Total height of the figure in pixels.
        width: Total width of the figure in pixels (optional).
        hovermode: Plotly hovermode to use ('x unified' is recommended for synced interaction across plots).
    
    Returns:
        go.Figure: The complete Plotly figure.
        
    Example:
        fig = plot_synced_timeseries(
            df=telemetry_df,
            x_col='Time',
            subplot_configs=[
                {
                    'title': 'Speed & RPM',
                    'y_label': 'Speed (km/h) / RPM',
                    'traces': [
                        {'y': 'Speed', 'name': 'Speed (km/h)', 'line': {'color': 'blue'}},
                        {'y': 'RPM', 'name': 'Engine RPM', 'line': {'color': 'red'}}
                    ]
                },
                {
                    'title': 'Throttle & Brake',
                    'y_label': 'Percentage (%)',
                    'traces': [
                        {'y': 'Throttle', 'name': 'Throttle %', 'line': {'color': 'green'}},
                        {'y': 'Brake', 'name': 'Brake %', 'line': {'color': 'orange'}}
                    ]
                }
            ],
            title='Telemetry Analysis: Lap 1'
        )
        fig.show()
    """
    num_subplots = len(subplot_configs)
    
    # Extract titles for each subplot
    subplot_titles = [config.get('title', f"Plot {i+1}") for i, config in enumerate(subplot_configs)]
    
    # Create the base figure with subplots
    fig = make_subplots(
        rows=num_subplots, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )
    
    # Add traces for each subplot
    for i, config in enumerate(subplot_configs):
        row = i + 1
        
        # Iterate over traces within the current subplot
        for trace_config in config.get('traces', []):
            y_col = trace_config.get('y')
            
            # Allow providing raw list/array if column name is not in DataFrame
            if isinstance(y_col, str) and y_col in df.columns:
                y_data = df[y_col]
            else:
                y_data = y_col
                
            name = trace_config.get('name', y_col if isinstance(y_col, str) else f"Trace {row}")
            mode = trace_config.get('mode', 'lines')
            line_opts = trace_config.get('line', None)
            marker_opts = trace_config.get('marker', None)
            
            trace = go.Scatter(
                x=df[x_col],
                y=y_data,
                name=name,
                mode=mode,
                line=line_opts,
                marker=marker_opts
            )
            fig.add_trace(trace, row=row, col=1)
            
        # Set Y-axis label if provided
        y_label = config.get('y_label', '')
        if y_label:
            fig.update_yaxes(title_text=y_label, row=row, col=1)

    # Configure layout for unified hover and sizing
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        hovermode=hovermode,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configure shared X-axis features (spikelines across all plots)
    # This enables a visual line extending across all subplots, useful for aligning events in time.
    fig.update_xaxes(
        showspikes=True, 
        spikemode="across", 
        spikedash="solid", 
        spikecolor="rgba(128, 128, 128, 0.5)", 
        spikethickness=1,
    )
    
    return fig
