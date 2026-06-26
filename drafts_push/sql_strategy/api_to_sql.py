import json
import os
from datetime import timedelta

import fastf1
import numpy as np
import psycopg

DATABASE_URL = os.getenv("DATABASE_URL")


def _parse_float(value):
    try:
        val = float(value)
    except TypeError:
        return None
    if np.isfinite(val):
        return val
    else:
        return None


def parse_session_object(session_obj):
    output = {}
    output["event_name"] = session_obj.session_info["Meeting"]["Name"]
    output["is_testing"] = False
    output["session_name"] = session_obj.name
    output["session_info"] = session_obj.session_info
    output["event_info"] = session_obj.event.to_dict()
    output["start_time"] = _parse_float(session_obj.session_start_time.total_seconds())
    output["t0_date"] = session_obj.t0_date
    output["total_laps"] = _parse_float(session_obj.total_laps)

    for col in ["session_info", "event_info"]:
        output[col] = json.dumps(output[col], default=str)

    return output


def parse_session_result(result):
    output = {}
    output["driver_number"] = result["DriverNumber"]
    output["broadcast_name"] = result["BroadcastName"]
    output["abbreviation"] = result["Abbreviation"]
    output["driver_id"] = result["DriverId"]
    output["team_name"] = result["TeamName"]
    output["team_color"] = "#" + result["TeamColor"].lower()
    output["team_id"] = result["TeamId"]
    output["first_name"] = result["FirstName"]
    output["last_name"] = result["LastName"]
    output["full_name"] = result["FullName"]
    output["headshot_url"] = result["HeadshotUrl"]
    output["country_code"] = result["CountryCode"]
    output["position"] = _parse_float(result["Position"])
    output["classified_position"] = str(result["ClassifiedPosition"])
    output["grid_position"] = _parse_float(result["GridPosition"])
    try:
        output["time_q1"] = _parse_float(result["Q1"].total_seconds())
    except TypeError:
        output["time_q1"] = None
    try:
        output["time_q2"] = _parse_float(result["Q2"].total_seconds())
    except TypeError:
        output["time_q1"] = None
    try:
        output["time_q3"] = _parse_float(result["Q3"].total_seconds())
    except TypeError:
        output["time_q1"] = None
    try:
        output["time_total"] = _parse_float(result["Time"].total_seconds())
    except TypeError:
        output["time_q1"] = None
    output["status"] = result["Status"]
    output["points"] = _parse_float(result["Points"])
    output["laps"] = _parse_float(result["Laps"])
    return output


def parse_session_weather(row, t0_date):
    output = {}
    output["timestamp_relative"] = _parse_float(row["Time"].total_seconds())
    if np.isfinite(output["timestamp_relative"]):
        output["timestamp_absolute"] = t0_date + timedelta(
            seconds=output["timestamp_relative"]
        )
    else:
        output["timestamp_absolute"] = None
    output["air_temperature"] = _parse_float(row["AirTemp"])
    output["track_temperature"] = _parse_float(row["TrackTemp"])
    output["humidity"] = _parse_float(row["Humidity"])
    output["pressure"] = _parse_float(row["Pressure"])
    output["rainfall"] = bool(row["Rainfall"])
    output["wind_direction"] = _parse_float(row["WindDirection"])
    output["wind_speed"] = _parse_float(row["WindSpeed"])
    return output


def parse_race_control_messages(row, t0_date):
    output = {}
    output["timestamp_absolute"] = row["Time"]
    output["timestamp_relative"] = _parse_float((row["Time"] - t0_date).total_seconds())
    output["category"] = row["Category"]
    output["status"] = row["Status"]
    output["scope"] = row["Scope"]
    output["flag"] = row["Flag"]
    output["message"] = row["Message"]
    output["racing_number"] = row["RacingNumber"]
    output["lap"] = _parse_float(row["Lap"])
    output["sector"] = _parse_float(row["Sector"])
    return output


def parse_session_status(row, t0_date):
    output = {}
    output["timestamp_relative"] = _parse_float(row["Time"].total_seconds())
    if np.isfinite(output["timestamp_relative"]):
        output["timestamp_absolute"] = t0_date + timedelta(
            seconds=output["timestamp_relative"]
        )
    else:
        output["timestamp_absolute"] = None
    output["status"] = str(row["Status"])
    return output


def parse_track_status(row, t0_date):
    output = {}
    output["timestamp_relative"] = _parse_float(row["Time"].total_seconds())
    if np.isfinite(output["timestamp_relative"]):
        output["timestamp_absolute"] = t0_date + timedelta(
            seconds=output["timestamp_relative"]
        )
    else:
        output["timestamp_absolute"] = None
    output["status"] = str(row["Status"])
    output["message"] = str(row["Message"])
    return output


def parse_circuit_features(row):
    output = {}
    try:
        output["number"] = int(row["Number"])
    except (TypeError, ValueError):
        output["number"] = None
    output["letter"] = row["Letter"]
    if not output["letter"]:
        output["letter"] = ""
    output["coordinate_x"] = _parse_float(row["X"])
    output["coordinate_y"] = _parse_float(row["Y"])
    output["angle"] = _parse_float(row["Angle"])
    output["distance"] = _parse_float(row["Distance"])
    return output


def parse_session_laps(row, t0_date):
    output = {}
    output["driver_number"] = row["DriverNumber"]
    output["driver_id"] = row["Driver"]
    output["team_name"] = row["Team"]
    output["lap_number"] = int(row["LapNumber"])
    output["timestamp_lap_start"] = _parse_float(row["LapStartTime"].total_seconds())
    output["timestamp_lap_end"] = _parse_float(row["Time"].total_seconds())
    output["timestamp_lap_start_absolute"] = row["LapStartDate"].isoformat()
    if output["timestamp_lap_start_absolute"].lower() == "nat":
        output["timestamp_lap_start_absolute"] = None
    if np.isfinite(output["timestamp_lap_end"]):
        output["timestamp_lap_end_absolute"] = t0_date + timedelta(
            seconds=output["timestamp_lap_end"]
        )
    else:
        output["timestamp_lap_end_absolute"] = None
    output["timestamp_pit_in"] = _parse_float(row["PitInTime"].total_seconds())
    output["timestamp_pit_out"] = _parse_float(row["PitOutTime"].total_seconds())
    output["timestamp_sector1"] = _parse_float(
        row["Sector1SessionTime"].total_seconds()
    )
    output["timestamp_sector2"] = _parse_float(
        row["Sector2SessionTime"].total_seconds()
    )
    output["timestamp_sector3"] = _parse_float(
        row["Sector3SessionTime"].total_seconds()
    )
    output["time_lap"] = _parse_float(row["LapTime"].total_seconds())
    output["time_sector1"] = _parse_float(row["Sector1Time"].total_seconds())
    output["time_sector2"] = _parse_float(row["Sector2Time"].total_seconds())
    output["time_sector3"] = _parse_float(row["Sector3Time"].total_seconds())
    output["speed_i1"] = _parse_float(row["SpeedI1"])
    output["speed_i2"] = _parse_float(row["SpeedI2"])
    output["speed_fl"] = _parse_float(row["SpeedFL"])
    output["speed_st"] = _parse_float(row["SpeedST"])
    output["is_personal_best"] = row["IsPersonalBest"] == True
    output["compound"] = row["Compound"]
    output["tyre_life"] = _parse_float(row["TyreLife"])
    output["fresh_tyre"] = row["FreshTyre"] == True
    output["stint"] = _parse_float(row["Stint"])
    output["track_status"] = row["TrackStatus"]
    output["position"] = _parse_float(row["Position"])
    output["is_deleted"] = row["Deleted"] == True
    output["deleted_reason"] = row["DeletedReason"]
    output["is_fastf1_generated"] = row["FastF1Generated"]
    output["is_accurate"] = row["IsAccurate"]

    return output


def parse_position_data(row):
    output = {}
    output["timestamp_relative"] = _parse_float(row["SessionTime"].total_seconds())
    output["timestamp_alternative"] = _parse_float(row["Time"].total_seconds())
    output["timestamp_absolute"] = row["Date"]
    output["status"] = row["Status"]
    output["source"] = row["Source"]
    output["coordinate_x"] = _parse_float(row["X"])
    output["coordinate_y"] = _parse_float(row["Y"])
    output["coordinate_z"] = _parse_float(row["Z"])
    return output


def parse_telemetry_data(row):
    output = {}
    output["timestamp_relative"] = _parse_float(row["SessionTime"].total_seconds())
    output["timestamp_alternative"] = _parse_float(row["Time"].total_seconds())
    output["timestamp_absolute"] = row["Date"]
    output["source"] = row["Source"]
    output["rpm"] = _parse_float(row["RPM"])
    output["speed"] = _parse_float(row["Speed"])
    output["gear"] = _parse_float(row["nGear"])
    output["throttle"] = _parse_float(row["Throttle"])
    output["brake"] = _parse_float(row["Brake"])
    output["drs"] = _parse_float(row["DRS"])
    return output


def process_session(year: int, round_number: int, session_number: int):
    session_id = f"Y{year}R{round_number:02d}S{session_number}"
    session_obj = fastf1.get_session(year, round_number, session_number)
    session_obj.load()

    add_session = parse_session_object(session_obj=session_obj)
    add_session["session_id"] = session_id
    add_session["year"] = year
    add_session["round_number"] = round_number
    add_session["session_number"] = session_number

    columns = list(add_session.keys())
    query = (
        "insert into race_session ( {} ) values ( {} ) on conflict do nothing".format(
            ",".join(columns), ",".join(["%s"] * len(columns))
        )
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [add_session[c] for c in columns])
            conn.commit()

    results = [
        {"session_id": session_id, **parse_session_result(x)}
        for x in session_obj.results.to_dict(orient="records")
    ]
    columns = list(results[0].keys())
    query = "insert into session_results ( {} ) values ( {} ) on conflict do nothing".format(
        ",".join(columns), ",".join(["%s"] * len(columns))
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in results])
            conn.commit()

    weather = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_session_weather(x, session_obj.t0_date),
        }
        for i, x in enumerate(
            session_obj.weather_data.sort_values("Time").to_dict(orient="records")
        )
    ]
    columns = list(weather[0].keys())
    query = "insert into session_weather ( {} ) values ( {} ) on conflict do nothing".format(
        ",".join(columns), ",".join(["%s"] * len(columns))
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in weather])
            conn.commit()

    race_control_messages = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_race_control_messages(x, session_obj.t0_date),
        }
        for i, x in enumerate(
            session_obj.race_control_messages.sort_values("Time").to_dict(
                orient="records"
            )
        )
    ]
    columns = list(race_control_messages[0].keys())
    query = "insert into race_control_messages ( {} ) values ( {} ) on conflict do nothing".format(
        ",".join(columns), ",".join(["%s"] * len(columns))
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                query, [[r[c] for c in columns] for r in race_control_messages]
            )
            conn.commit()

    session_status = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_session_status(x, session_obj.t0_date),
        }
        for i, x in enumerate(
            session_obj.session_status.sort_values("Time").to_dict(orient="records")
        )
    ]
    columns = list(session_status[0].keys())
    query = (
        "insert into session_status ( {} ) values ( {} ) on conflict do nothing".format(
            ",".join(columns), ",".join(["%s"] * len(columns))
        )
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in session_status])
            conn.commit()

    track_status = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_track_status(x, session_obj.t0_date),
        }
        for i, x in enumerate(
            session_obj.track_status.sort_values("Time").to_dict(orient="records")
        )
    ]
    columns = list(track_status[0].keys())
    query = (
        "insert into track_status ( {} ) values ( {} ) on conflict do nothing".format(
            ",".join(columns), ",".join(["%s"] * len(columns))
        )
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in track_status])
            conn.commit()

    laps = [
        {
            "session_id": session_id,
            **parse_session_laps(x, session_obj.t0_date),
        }
        for x in (session_obj.laps.to_dict(orient="records"))
    ]
    columns = list(laps[0].keys())
    query = (
        "insert into session_laps ( {} ) values ( {} ) on conflict do nothing".format(
            ",".join(columns), ",".join(["%s"] * len(columns))
        )
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in laps])
            conn.commit()

    circuit_info = session_obj.get_circuit_info()
    rotation = circuit_info.rotation
    circuit_features = []
    circuit_features += [
        {
            "session_id": session_id,
            "feature_type": "corner",
            "circuit_rotation": rotation,
            **parse_circuit_features(x),
        }
        for x in circuit_info.corners.to_dict(orient="records")
    ]
    circuit_features += [
        {
            "session_id": session_id,
            "feature_type": "marshal_light",
            "circuit_rotation": rotation,
            **parse_circuit_features(x),
        }
        for x in circuit_info.marshal_lights.to_dict(orient="records")
    ]
    circuit_features += [
        {
            "session_id": session_id,
            "feature_type": "marshal_sector",
            "circuit_rotation": rotation,
            **parse_circuit_features(x),
        }
        for x in circuit_info.marshal_sectors.to_dict(orient="records")
    ]
    columns = list(circuit_features[0].keys())
    query = "insert into circuit_features ( {} ) values ( {} ) on conflict do nothing".format(
        ",".join(columns), ",".join(["%s"] * len(columns))
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in circuit_features])
            conn.commit()

    position_data = [
        {
            "session_id": session_id,
            "driver_number": driver,
            **parse_position_data(x),
        }
        for driver, df in session_obj.pos_data.items()
        for x in df.to_dict(orient="records")
    ]
    columns = list(position_data[0].keys())
    query = (
        "insert into position_data ( {} ) values ( {} ) on conflict do nothing".format(
            ",".join(columns), ",".join(["%s"] * len(columns))
        )
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in position_data])
            conn.commit()

    telemetry_data = [
        {
            "session_id": session_id,
            "driver_number": driver,
            **parse_telemetry_data(x),
        }
        for driver, df in session_obj.car_data.items()
        for x in df.to_dict(orient="records")
    ]
    columns = list(telemetry_data[0].keys())
    query = (
        "insert into telemetry_data ( {} ) values ( {} ) on conflict do nothing".format(
            ",".join(columns), ",".join(["%s"] * len(columns))
        )
    )
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, [[r[c] for c in columns] for r in telemetry_data])
            conn.commit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("round_number", type=int)
    parser.add_argument("session_number", type=int)
    args = parser.parse_args()
    process_session(
        year=args.year,
        round_number=args.round_number,
        session_number=args.session_number,
    )
