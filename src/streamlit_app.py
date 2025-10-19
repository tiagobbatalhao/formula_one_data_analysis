from datetime import timedelta
from requests import session
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import math
import sklearn.linear_model
from data_engineering.gold_layer import HistoricalSessions, SessionLaps, SessionResults, SessionMetadata, SessionLaps

st.set_page_config(layout="wide")

def load_historical_sessions():
    dataset = HistoricalSessions()
    return dataset.read(force=True)

def load_session_results(year: int):
    dataset = SessionResults(year)
    return dataset.read(force=True)

def load_session_metadata(year: int):
    dataset = SessionMetadata(year)
    return dataset.read(force=True)

def load_session_laps(year: int):
    dataset = SessionLaps(year)
    return dataset.read(force=True)

def tab_session_result(session_info):
    selected_year = session_info['year']
    selected_session_id = session_info['session_id']
    df = load_session_results(selected_year)
    df = df[df['session_id']==selected_session_id]
    df = df.sort_values(by=['position', 'driver_number']).reset_index(drop=True)
    columns = []
    columns.append((
        'position',
        st.column_config.NumberColumn(label='Position', width=None)
    ))
    columns.append((
        'driver_number',
        st.column_config.NumberColumn(label='Number', width=None)
    ))
    columns.append((
        'driver_broadcast_name',
        st.column_config.TextColumn(label='Driver', width=None)
    ))
    columns.append((
        'team_name',
        st.column_config.TextColumn(label='Team', width=None)
    ))
    if session_info['session_name'] in ['Qualifying', 'Sprint Qualifying']:
        lap_times = [
            x for x in df[['time_q1', 'time_q2', 'time_q3']].values.flatten()
            if pd.notna(x)
        ]
        bounds = dict(min_value=min(lap_times), max_value=max(lap_times))
        
        columns.append((
            'time_q1',
            st.column_config.ProgressColumn(label='Time (Q1)', width=None, format="%.3f", **bounds)
        ))
        columns.append((
            'time_q2',
            st.column_config.ProgressColumn(label='Time (Q2)', width=None, format="%.3f", **bounds)
        ))
        columns.append((
            'time_q3',
            st.column_config.ProgressColumn(label='Time (Q3)', width=None, format="%.3f", **bounds)
        ))
    if session_info['session_name'] in ['Race', 'Sprint']:
        columns.append((
            'status',
            st.column_config.TextColumn(label='Status', width=None)
        ))
        columns.append((
            'points',
            st.column_config.ProgressColumn(label='Points', width=None, min_value=0, max_value=25, format='%d')
        ))

    st.dataframe(
        df, column_config={x[0]: x[1] for x in columns},
        column_order=[x[0] for x in columns], hide_index=True)
    st.dataframe(df,)

def tab_session_laps(session_info):
    selected_year = session_info['year']
    selected_session_id = session_info['session_id']
    df = load_session_laps(selected_year)
    df = df[df['session_id']==selected_session_id]
    df = df[df['is_accurate']]
    df = df.dropna(subset=['timestamp_lap_start', 'time_lap'])
    df['timestamp_lap_end'] = df.apply(
        lambda row: row['timestamp_lap_start'] + timedelta(seconds=row['time_lap']),
        axis=1,
    )
    df = df.dropna(subset=['timestamp_lap_start', 'time_lap'])
    df = df.sort_values(by=['timestamp_lap_end'])

    min_lap_time = df['time_lap'].min()

    plot_data = []
    for driver in sorted(df['driver_number'].unique()):
        this = df[df['driver_number']==driver]
        if len(this) == 0:
            continue
        plot_data += [
            go.Scatter(
                x=this['timestamp_lap_end'],
                y=this['time_lap'],
                mode='markers',
                name=this.iloc[0]['driver_name'],
            )
        ]
    
    plot_layout = dict(
        height=600,
        title="Laptime history",
        yaxis=dict(range=(
            math.floor(min_lap_time * 0.99 / 0.1) * 0.1,
            math.ceil(min_lap_time * 1.11 / 0.1) * 0.1,
        ))
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

def tab_session_race_trace(session_info):
    if session_info['session_name'] not in ['Race', 'Sprint']:
        return
    selected_year = session_info['year']
    selected_session_id = session_info['session_id']
    df = load_session_laps(selected_year)
    df = df[df['session_id']==selected_session_id]
    to_fit = df['timing_end_sector3'].notna() & df['lap_number'].notna()
    df = df[to_fit]
    regressor = sklearn.linear_model.QuantileRegressor(quantile=0.2).fit(
        df.loc[:, ['lap_number']],
        df.loc[:, 'timing_end_sector3']
    )
    df['expected'] = regressor.predict(df[['lap_number']])
    st.dataframe(df[['driver_number', 'lap_number', 'timing_end_sector3', 'expected']])

    plot_data = []
    for driver in sorted(df['driver_number'].unique()):
        this = df[df['driver_number']==driver]
        if len(this) == 0:
            continue
        plot_data += [
            go.Scatter(
                x=this['lap_number'],
                y=(this['timing_end_sector3']-this['expected']) * (-1),
                mode='lines+markers',
                name=this.iloc[0]['driver_name'],
            )
        ]
    
    plot_layout = dict(
        height=600,
        title="Race trace (delta to average)",
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

def main():
    st.title("Formula 1 Data Visualization")

    # Load historical sessions for selection
    st.sidebar.header("Select Session")
    years = list(range(1950, 2026))[::-1]
    selected_year = st.sidebar.selectbox("Year", years)
    sessions_for_year = load_session_metadata(year=selected_year)
    round_names = sessions_for_year.drop_duplicates(subset=["event_name"])['event_name'].tolist()
    selected_round = st.sidebar.selectbox("Round", round_names)
    event_info = sessions_for_year[
        sessions_for_year["event_name"] == selected_round
    ]
    selected_session_name = st.sidebar.selectbox("Session", event_info["session_name"].tolist())
    session_info = event_info[
        event_info["session_name"] == selected_session_name
    ].iloc[0].to_dict()

    st.markdown(session_info)
    st.markdown("Event name: {}".format(session_info['meeting_name']))
    st.markdown("Session name: {}".format(session_info['session_name']))

    # selected_session_id = "Y{:04d}R{:02d}S{:01d}".format(
    #     selected_year, self.round_number, self.session_number
    # )

    # # Get session metadata for selected year
    # session_metadata = load_session_metadata(selected_year)
    # if session_metadata is None or session_metadata.empty:
    #     st.error("No session metadata available for selected year.")
    #     return

    # # Display session metadata summary
    # st.subheader(f"Session Metadata for {selected_year}")
    # st.dataframe(session_metadata)

    # # Display session results for selected year
    # session_results = load_session_results(selected_year)
    # if session_results is None or session_results.empty:
    #     st.warning("No session results available for selected year.")
    # else:
    #     st.subheader(f"Session Results for {selected_year}")
    #     st.dataframe(session_results)

    tabs = st.tabs(["Result", "Lap history", "Race trace"])

    with tabs[0]:
        tab_session_result(session_info)
    with tabs[1]:
        tab_session_laps(session_info)
    with tabs[2]:
        tab_session_race_trace(session_info)

if __name__ == "__main__":
    main()
