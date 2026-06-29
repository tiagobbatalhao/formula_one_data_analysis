import json
import os
from datetime import timedelta

import fastf1
import numpy as np
import psycopg
from loguru import logger

DATABASE_URL = "postgresql://f1user@localhost:5435/f1database"


def write_to_database(query, data):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, data)
        conn.commit()


def create_insert_query(table_name, columns, conflict=None):
    query = """
        insert into {table_name} ( {columns} )
        values ( {values} )
        on conflict {conflict} ;
    """.format(
        table_name=table_name,
        columns=",".join(columns),
        values=",".join(["%s"] * len(columns)),
        conflict=(
            " do nothing"
            if conflict is None
            else "( {keys} ) do update set {update_columns}".format(
                keys=" , ".join(conflict[0]),
                update_columns=" , ".join(
                    [f" {col} = EXCLUDED.{col} " for col in conflict[1]]
                ),
            )
        ),
    )
    return query


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
    output["session_name"] = session_obj.name
    output["session_data"] = session_obj.session_info
    output["event_data"] = session_obj.event.to_dict()
    output["start_time"] = _parse_float(session_obj.session_start_time.total_seconds())
    try:
        output["t0_date"] = session_obj.t0_date
    except fastf1.exceptions.DataNotLoadedError:
        pass
    output["total_laps"] = _parse_float(session_obj.total_laps)

    for col in ["session_data", "event_data"]:
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
    if (t0_date is not None) and np.isfinite(output["timestamp_relative"]):
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
    if t0_date is not None:
        output["timestamp_relative"] = _parse_float(
            (row["Time"] - t0_date).total_seconds()
        )
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
    if (t0_date is not None) and np.isfinite(output["timestamp_relative"]):
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
    if (t0_date is not None) and np.isfinite(output["timestamp_relative"]):
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
    if (t0_date is not None) and np.isfinite(output["timestamp_lap_end"]):
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


def process_session(year: int, event_id: str, session_number: int, telemetry: bool):
    session_id = f"Y{year}{event_id}S{session_number}"
    if (len(event_id) == 3) and (event_id[0] == "R"):
        session_obj = fastf1.get_session(year, int(event_id[1:]), session_number)
        event_type = "race_weekend"
    elif (len(event_id) == 3) and (event_id[0] == "T"):
        session_obj = fastf1.get_testing_session(
            year, int(event_id[1:]), session_number
        )
        event_type = "race_weekend"
    else:
        raise ValueError()
    session_obj.load(
        laps=True,
        weather=True,
        messages=True,
        telemetry=telemetry or True,
    )

    add_session = parse_session_object(session_obj=session_obj)
    add_session["session_id"] = session_id
    add_session["year"] = year
    add_session["event_id"] = event_id
    add_session["session_number"] = session_number
    add_session["event_type"] = event_type

    logger.info("Running session_information...")
    columns = list(add_session.keys())
    primary_key = ["session_id"]
    query = create_insert_query(
        table_name="session_information",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[add_session[c] for c in columns]])
    try:
        reference_timestamp = session_obj.t0_date
    except fastf1.exceptions.DataNotLoadedError:
        reference_timestamp = None

    logger.info("Running session_results...")
    results = [
        {"session_id": session_id, **parse_session_result(x)}
        for x in session_obj.results.to_dict(orient="records")
    ]
    columns = list(results[0].keys())
    primary_key = ['session_id', 'driver_number']
    query = create_insert_query(
        table_name="session_results",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[r[c] for c in columns] for r in results])

    logger.info("Running session_weather...")
    weather = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_session_weather(x, reference_timestamp),
        }
        for i, x in enumerate(
            session_obj.weather_data.sort_values("Time").to_dict(orient="records")
        )
    ]
    columns = list(weather[0].keys())
    primary_key = ["session_id", "counter"]
    query = create_insert_query(
        table_name="session_weather",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[r[c] for c in columns] for r in weather])

    logger.info("Running session_messages...")
    race_control_messages = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_race_control_messages(x, reference_timestamp),
        }
        for i, x in enumerate(
            session_obj.race_control_messages.sort_values("Time").to_dict(
                orient="records"
            )
        )
    ]
    columns = list(race_control_messages[0].keys())
    primary_key = ["session_id", "counter"]
    query = create_insert_query(
        table_name="session_messages",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[r[c] for c in columns] for r in race_control_messages])

    logger.info("Running session_status...")
    session_status = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_session_status(x, reference_timestamp),
        }
        for i, x in enumerate(
            session_obj.session_status.sort_values("Time").to_dict(orient="records")
        )
    ]
    columns = list(session_status[0].keys())
    primary_key = ["session_id", "counter"]
    query = create_insert_query(
        table_name="session_status",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[r[c] for c in columns] for r in session_status])

    logger.info("Running session_track_status...")
    track_status = [
        {
            "session_id": session_id,
            "counter": i + 1,
            **parse_track_status(x, reference_timestamp),
        }
        for i, x in enumerate(
            session_obj.track_status.sort_values("Time").to_dict(orient="records")
        )
    ]
    columns = list(track_status[0].keys())
    primary_key = ["session_id", "counter"]
    query = create_insert_query(
        table_name="session_track_status",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[r[c] for c in columns] for r in track_status])

    logger.info("Running session_laps...")
    laps = [
        {
            "session_id": session_id,
            **parse_session_laps(x, reference_timestamp),
        }
        for x in (session_obj.laps.to_dict(orient="records"))
    ]
    columns = list(laps[0].keys())
    primary_key = ["session_id", "driver_number", "lap_number"]
    query = create_insert_query(
        table_name="session_laps",
        columns=columns,
        conflict=(primary_key, [c for c in columns if c not in primary_key]),
    )
    write_to_database(query, [[r[c] for c in columns] for r in laps])

    try:
        logger.info("Running circuit_features...")
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
        primary_key = ["session_id", "feature_type", "number", "letter"]
        query = create_insert_query(
            table_name="circuit_features",
            columns=columns,
            conflict=(primary_key, [c for c in columns if c not in primary_key]),
        )
        write_to_database(query, [[r[c] for c in columns] for r in circuit_features])
    except Exception as e:
        logger.error(f"Error in circuit_info: {e}")

    if telemetry:
        for driver, df in session_obj.pos_data.items():
            logger.info(f"Running telemetry_position for driver {driver}...")
            position_data = [
                {
                    "session_id": session_id,
                    "driver_number": driver,
                    "counter": i + 1,
                    **parse_position_data(x),
                }
                for i, x in enumerate(df.to_dict(orient="records"))
            ]
            columns = list(position_data[0].keys())
            primary_key = ["session_id", "driver_number", "counter"]
            query = create_insert_query(
                table_name="telemetry_position",
                columns=columns,
                conflict=(primary_key, [c for c in columns if c not in primary_key]),
            )
            write_to_database(query, [[r[c] for c in columns] for r in position_data])

    if telemetry:
        for driver, df in session_obj.car_data.items():
            logger.info(f"Running telemetry_car for driver {driver}...")
            telemetry_data = [
                {
                    "session_id": session_id,
                    "driver_number": driver,
                    "counter": i + 1,
                    **parse_telemetry_data(x),
                }
                for i, x in enumerate(df.to_dict(orient="records"))
            ]
            columns = list(telemetry_data[0].keys())
            primary_key = ["session_id", "driver_number", "counter"]
            query = create_insert_query(
                table_name="telemetry_car",
                columns=columns,
                conflict=(primary_key, [c for c in columns if c not in primary_key]),
            )
            write_to_database(query, [[r[c] for c in columns] for r in telemetry_data])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("event_id", type=str)
    parser.add_argument("session_number", type=int)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()
    process_session(
        year=args.year,
        event_id=args.event_id,
        session_number=args.session_number,
        telemetry=bool(args.telemetry),
    )
