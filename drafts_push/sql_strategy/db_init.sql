DROP TABLE race_session;
DROP TABLE session_results;
DROP TABLE session_weather;
DROP TABLE race_control_messages;
DROP TABLE session_laps;
DROP TABLE session_status;
DROP TABLE track_status;
DROP TABLE circuit_features;
DROP TABLE position_data;
DROP TABLE telemetry_data;



CREATE TABLE IF NOT EXISTS race_session (
    session_id varchar(127),
    year integer,
    round_number integer,
    session_number integer,
    is_testing bool,
    event_name varchar(127),
    session_name varchar(127),
    session_info json,
    event_info json,
    start_time float,
    t0_date timestamp with time zone,
    total_laps float,
    PRIMARY KEY (session_id)
);

CREATE TABLE IF NOT EXISTS session_results (
    session_id varchar(127),
    driver_number varchar(127),
    broadcast_name varchar(127),
    abbreviation varchar(127),
    driver_id varchar(127),
    team_name varchar(127),
    team_color varchar(127),
    team_id varchar(127),
    first_name varchar(127),
    last_name varchar(127),
    full_name varchar(127),
    headshot_url text,
    country_code varchar(127),
    position float,
    classified_position varchar(127),
    grid_position float,
    time_q1 float,
    time_q2 float,
    time_q3 float,
    time_total float,
    status varchar(127),
    points float,
    laps float,
    PRIMARY KEY (session_id, driver_number)
);

CREATE TABLE IF NOT EXISTS session_weather (
    session_id varchar(127),
    counter integer,
    timestamp_relative float,
    timestamp_absolute timestamp with time zone,
    air_temperature float,
    track_temperature float,
    humidity float,
    pressure float,
    rainfall bool,
    wind_direction float,
    wind_speed float,
    PRIMARY KEY (session_id, counter)
);

CREATE TABLE IF NOT EXISTS race_control_messages (
    session_id varchar(127),
    counter integer,
    timestamp_relative float,
    timestamp_absolute timestamp with time zone,
    category varchar(127),
    status varchar(127),
    scope varchar(127),
    flag varchar(127),
    message text,
    racing_number varchar(127),
    sector float,
    lap float,
    PRIMARY KEY (session_id, counter)
);

CREATE TABLE IF NOT EXISTS session_status (
    session_id varchar(127),
    counter integer,
    timestamp_relative float,
    timestamp_absolute timestamp with time zone,
    status varchar(127),
    PRIMARY KEY (session_id, counter)
);

CREATE TABLE IF NOT EXISTS track_status (
    session_id varchar(127),
    counter integer,
    timestamp_relative float,
    timestamp_absolute timestamp with time zone,
    status varchar(127),
    message text,
    PRIMARY KEY (session_id, counter)
);

CREATE TABLE IF NOT EXISTS circuit_features (
    session_id varchar(127),
    feature_type varchar(127),
    number integer,
    letter varchar(127),
    circuit_rotation float,
    coordinate_x float,
    coordinate_y float,
    angle float,
    distance float,
    PRIMARY KEY (session_id, feature_type, number, letter)
);

CREATE TABLE IF NOT EXISTS session_laps (
    session_id varchar(127),
    driver_number varchar(127),
    driver_id varchar(127),
    team_name varchar(127),
    lap_number integer,
    timestamp_lap_start float,
    timestamp_lap_end float,
    timestamp_lap_start_absolute timestamp with time zone,
    timestamp_lap_end_absolute timestamp with time zone,
    timestamp_pit_in float,
    timestamp_pit_out float,
    timestamp_sector1 float,
    timestamp_sector2 float,
    timestamp_sector3 float,
    time_lap float,
    time_sector1 float,
    time_sector2 float,
    time_sector3 float,
    speed_i1 float,
    speed_i2 float,
    speed_fl float,
    speed_st float,
    is_personal_best boolean,
    compound varchar(127),
    tyre_life float,
    fresh_tyre boolean,
    stint float,
    track_status varchar(127),
    position float,
    is_deleted boolean,
    deleted_reason text,
    is_fastf1_generated boolean,
    is_accurate boolean,
    PRIMARY KEY (session_id, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS position_data (
    session_id varchar(127),
    driver_number varchar(127),
    timestamp_relative float,
    timestamp_absolute timestamp with time zone,
    timestamp_alternative float,
    status varchar(127),
    source varchar(127),
    coordinate_x float,
    coordinate_y float,
    coordinate_z float,
    PRIMARY KEY (session_id, driver_number, timestamp_relative)
);

CREATE TABLE IF NOT EXISTS telemetry_data (
    session_id varchar(127),
    driver_number varchar(127),
    timestamp_relative float,
    timestamp_absolute timestamp with time zone,
    timestamp_alternative float,
    source varchar(127),
    rpm float,
    speed float,
    gear float,
    throttle float,
    brake float,
    drs float,
    PRIMARY KEY (session_id, driver_number, timestamp_relative)
);
