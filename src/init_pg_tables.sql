CREATE TABLE IF NOT EXISTS historical_sessions (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    round_number INTEGER NOT NULL,
    event_name VARCHAR NOT NULL,
    event_format VARCHAR NOT NULL,
    event_date DATE NOT NULL,
    country VARCHAR NOT NULL,
    location VARCHAR NOT NULL,
    official_event_name VARCHAR,
    has_api_support BOOLEAN NOT NULL,
    session_number INTEGER NOT NULL,
    session_name VARCHAR NOT NULL,
    session_time_utc TIMESTAMP NOT NULL,
    PRIMARY KEY (session_id)
);

CREATE TABLE IF NOT EXISTS session_metadata (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    session_name VARCHAR NOT NULL,
    f1_api_support BOOLEAN NOT NULL,
    timestamp_reference TIMESTAMP NOT NULL,
    session_scheduled_time TIMESTAMP NOT NULL,
    timing_start DOUBLE PRECISION NOT NULL,
    total_laps INTEGER,
    driver_list INTEGER[] NOT NULL,
    meeting_key INTEGER NOT NULL,
    meeting_name VARCHAR NOT NULL,
    meeting_official_name VARCHAR NOT NULL,
    meeting_location VARCHAR NOT NULL,
    meeting_number INTEGER NOT NULL,
    country_key INTEGER NOT NULL,
    country_code VARCHAR NOT NULL,
    country_name VARCHAR NOT NULL,
    circuit_key INTEGER NOT NULL,
    circuit_short_name VARCHAR NOT NULL,
    round_number INTEGER NOT NULL,
    event_country VARCHAR NOT NULL,
    event_location VARCHAR NOT NULL,
    event_name VARCHAR NOT NULL,
    event_official_name VARCHAR NOT NULL,
    event_date DATE NOT NULL,
    event_format VARCHAR NOT NULL,
    type VARCHAR NOT NULL,
    scheduled_time_utc TIMESTAMP NOT NULL,
    scheduled_time_local TIMESTAMP NOT NULL,
    PRIMARY KEY (session_id)
);

CREATE TABLE IF NOT EXISTS session_results (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    driver_id VARCHAR,
    driver_number INTEGER NOT NULL,
    driver_broadcast_name VARCHAR NOT NULL,
    driver_abbreviation VARCHAR NOT NULL,
    driver_first_name VARCHAR NOT NULL,
    driver_last_name VARCHAR NOT NULL,
    driver_full_name VARCHAR NOT NULL,
    driver_headshot_url VARCHAR,
    team_color VARCHAR NOT NULL,
    position INTEGER,
    classified_position INTEGER,
    grid_position INTEGER,
    time_q1 DOUBLE PRECISION,
    time_q2 DOUBLE PRECISION,
    time_q3 DOUBLE PRECISION,
    time DOUBLE PRECISION,
    status VARCHAR,
    points INTEGER,
    PRIMARY KEY (session_id, driver_number)
);

CREATE TABLE IF NOT EXISTS session_laps (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    driver_number INTEGER NOT NULL,
    driver_name VARCHAR NOT NULL,
    driver_team VARCHAR NOT NULL,
    lap_number INTEGER NOT NULL,
    stint INTEGER,
    timestamp_lap_start TIMESTAMP NOT NULL,
    timing_start_lap DOUBLE PRECISION,
    timing_end_lap DOUBLE PRECISION,
    timing_end_sector1 DOUBLE PRECISION,
    timing_end_sector2 DOUBLE PRECISION,
    timing_end_sector3 DOUBLE PRECISION,
    timing_pit_out DOUBLE PRECISION,
    timing_pit_in DOUBLE PRECISION,
    time_lap DOUBLE PRECISION,
    time_sector1 DOUBLE PRECISION,
    time_sector2 DOUBLE PRECISION,
    time_sector3 DOUBLE PRECISION,
    tyre_compound VARCHAR,
    tyre_life INTEGER,
    speed_i1 DOUBLE PRECISION,
    speed_i2 DOUBLE PRECISION,
    speed_fl DOUBLE PRECISION,
    speed_st DOUBLE PRECISION,
    is_personal_best BOOLEAN NOT NULL,
    track_status INTEGER,
    position INTEGER,
    deleted_reason VARCHAR,
    is_accurate BOOLEAN NOT NULL,
    is_fastf1_generated BOOLEAN NOT NULL,
    PRIMARY KEY (session_id, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS session_lap_timings (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    driver_number INTEGER NOT NULL,
    lap_number INTEGER NOT NULL,
    timestamp_start TIMESTAMP,
    timestamp_pitout TIMESTAMP,
    timestamp_sector1 TIMESTAMP,
    timestamp_sector2 TIMESTAMP,
    timestamp_pitin TIMESTAMP,
    timestamp_sector3 TIMESTAMP,
    timestamp_end TIMESTAMP,
    PRIMARY KEY (session_id, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS session_weather (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timing_from_session DOUBLE PRECISION NOT NULL,
    rainfall BOOLEAN NOT NULL,
    air_temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    track_temperature DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS session_race_control_messages (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timing_from_session DOUBLE PRECISION NOT NULL,
    category VARCHAR NOT NULL,
    message VARCHAR,
    status VARCHAR,
    flag VARCHAR,
    scope VARCHAR,
    sector INTEGER,
    driver_number INTEGER,
    lap_number INTEGER
);

CREATE TABLE IF NOT EXISTS session_track_status (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timing_from_session DOUBLE PRECISION NOT NULL,
    status INTEGER NOT NULL,
    message VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS circuit_markers (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    annotation_type VARCHAR NOT NULL,
    number INTEGER NOT NULL,
    letter VARCHAR,
    coordinate_x DOUBLE PRECISION NOT NULL,
    coordinate_y DOUBLE PRECISION NOT NULL,
    angle DOUBLE PRECISION NOT NULL,
    distance DOUBLE PRECISION NOT NULL,
    rotation DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_car (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    driver_number INTEGER NOT NULL,
    lap_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timing_from_session DOUBLE PRECISION NOT NULL,
    timing_from_lap DOUBLE PRECISION NOT NULL,
    track_status INTEGER,
    rpm DOUBLE PRECISION NOT NULL,
    speed DOUBLE PRECISION NOT NULL,
    ngear INTEGER NOT NULL,
    brake DOUBLE PRECISION NOT NULL,
    drs INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_pos (
    year INTEGER NOT NULL,
    session_id VARCHAR NOT NULL,
    driver_number INTEGER NOT NULL,
    lap_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timing_from_session DOUBLE PRECISION NOT NULL,
    timing_from_lap DOUBLE PRECISION NOT NULL,
    track_status INTEGER,
    coordinate_x DOUBLE PRECISION NOT NULL,
    coordinate_y DOUBLE PRECISION NOT NULL,
    coordinate_z DOUBLE PRECISION NOT NULL,
    position_status VARCHAR NOT NULL
);
