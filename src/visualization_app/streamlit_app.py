import streamlit as st
from data_loader import load_session_metadata
from tab_session_laps import tab_session_laps
from tab_session_race_trace import tab_session_race_trace
from tab_session_result import tab_session_result

st.set_page_config(layout="wide")


def main():

    # Load historical sessions for selection
    st.sidebar.header("Select Session")
    years = list(range(1950, 2027))[::-1]
    selected_year = st.sidebar.selectbox("Year", years)
    sessions_for_year = load_session_metadata(year=selected_year)
    round_names = sessions_for_year.drop_duplicates(subset=["event_name"])[
        "event_name"
    ].tolist()
    selected_round = st.sidebar.selectbox("Round", round_names)
    event_info = sessions_for_year[sessions_for_year["event_name"] == selected_round]
    selected_session_name = st.sidebar.selectbox(
        "Session", event_info["session_name"].tolist()
    )
    session_info = (
        event_info[event_info["session_name"] == selected_session_name]
        .iloc[0]
        .to_dict()
    )

    st.title(
        "{} - {}".format(session_info["meeting_name"], session_info["session_name"])
    )

    tabs = st.tabs(["Result", "Lap history", "Race trace"])

    with tabs[0]:
        tab_session_result(session_info)
    with tabs[1]:
        tab_session_laps(session_info)
    with tabs[2]:
        tab_session_race_trace(session_info)


if __name__ == "__main__":
    main()
