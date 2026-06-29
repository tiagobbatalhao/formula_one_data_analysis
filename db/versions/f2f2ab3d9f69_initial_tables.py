"""initial tables

Revision ID: f2f2ab3d9f69
Revises:
Create Date: 2026-06-26 12:52:20.719880

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f2f2ab3d9f69"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "session_information",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("year", sa.Integer()),
        sa.Column("event_id", sa.String(length=32)),
        sa.Column("session_number", sa.Integer()),
        sa.Column("event_type", sa.String(length=32)),
        sa.Column("event_name", sa.Text()),
        sa.Column("session_name", sa.Text()),
        sa.Column("session_data", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("event_data", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("start_time", sa.Float()),
        sa.Column("t0_date", sa.DateTime(timezone=True)),
        sa.Column("total_laps", sa.Float()),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index(
        "idx_session_information", "session_information", ["year"], unique=False
    )

    op.create_table(
        "session_results",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("driver_number", sa.String(length=32)),
        sa.Column("broadcast_name", sa.String(length=128)),
        sa.Column("abbreviation", sa.String(length=128)),
        sa.Column("driver_id", sa.String(length=128)),
        sa.Column("team_name", sa.String(length=128)),
        sa.Column("team_color", sa.String(length=128)),
        sa.Column("team_id", sa.String(length=128)),
        sa.Column("first_name", sa.String(length=128)),
        sa.Column("last_name", sa.String(length=128)),
        sa.Column("full_name", sa.String(length=128)),
        sa.Column("headshot_url", sa.Text()),
        sa.Column("country_code", sa.String(length=32)),
        sa.Column("position", sa.Float()),
        sa.Column("classified_position", sa.String(length=32)),
        sa.Column("grid_position", sa.Float()),
        sa.Column("time_q1", sa.Float()),
        sa.Column("time_q2", sa.Float()),
        sa.Column("time_q3", sa.Float()),
        sa.Column("time_total", sa.Float()),
        sa.Column("status", sa.String(length=128)),
        sa.Column("points", sa.Float()),
        sa.Column("laps", sa.Float()),
        sa.PrimaryKeyConstraint("session_id", "driver_number"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_session_results", "session_results", ["session_id"], unique=False
    )

    op.create_table(
        "session_weather",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("counter", sa.Integer()),
        sa.Column("timestamp_relative", sa.Float()),
        sa.Column("timestamp_absolute", sa.DateTime(timezone=True)),
        sa.Column("air_temperature", sa.Float()),
        sa.Column("track_temperature", sa.Float()),
        sa.Column("humidity", sa.Float()),
        sa.Column("pressure", sa.Float()),
        sa.Column("rainfall", sa.Boolean()),
        sa.Column("wind_direction", sa.Float()),
        sa.Column("wind_speed", sa.Float()),
        sa.PrimaryKeyConstraint("session_id", "counter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_session_weather", "session_weather", ["session_id"], unique=False
    )

    op.create_table(
        "session_messages",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("counter", sa.Integer()),
        sa.Column("timestamp_relative", sa.Float()),
        sa.Column("timestamp_absolute", sa.DateTime(timezone=True)),
        sa.Column("category", sa.String(length=128)),
        sa.Column("status", sa.String(length=128)),
        sa.Column("scope", sa.String(length=128)),
        sa.Column("flag", sa.String(length=128)),
        sa.Column("message", sa.Text()),
        sa.Column("racing_number", sa.String(length=128)),
        sa.Column("sector", sa.Float()),
        sa.Column("lap", sa.Float()),
        sa.PrimaryKeyConstraint("session_id", "counter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_session_messages", "session_messages", ["session_id"], unique=False
    )

    op.create_table(
        "session_status",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("counter", sa.Integer()),
        sa.Column("timestamp_relative", sa.Float()),
        sa.Column("timestamp_absolute", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(length=128)),
        sa.PrimaryKeyConstraint("session_id", "counter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_session_status", "session_status", ["session_id"], unique=False
    )

    op.create_table(
        "session_track_status",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("counter", sa.Integer()),
        sa.Column("timestamp_relative", sa.Float()),
        sa.Column("timestamp_absolute", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(length=128)),
        sa.Column("message", sa.Text()),
        sa.PrimaryKeyConstraint("session_id", "counter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_session_track_status", "session_track_status", ["session_id"], unique=False
    )

    op.create_table(
        "session_laps",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("driver_number", sa.String(length=32)),
        sa.Column("driver_id", sa.String(length=128)),
        sa.Column("team_name", sa.String(length=128)),
        sa.Column("lap_number", sa.Integer()),
        sa.Column("timestamp_lap_start", sa.Float()),
        sa.Column("timestamp_lap_end", sa.Float()),
        sa.Column("timestamp_lap_start_absolute", sa.DateTime(timezone=True)),
        sa.Column("timestamp_lap_end_absolute", sa.DateTime(timezone=True)),
        sa.Column("timestamp_pit_in", sa.Float()),
        sa.Column("timestamp_pit_out", sa.Float()),
        sa.Column("timestamp_sector1", sa.Float()),
        sa.Column("timestamp_sector2", sa.Float()),
        sa.Column("timestamp_sector3", sa.Float()),
        sa.Column("time_lap", sa.Float()),
        sa.Column("time_sector1", sa.Float()),
        sa.Column("time_sector2", sa.Float()),
        sa.Column("time_sector3", sa.Float()),
        sa.Column("speed_i1", sa.Float()),
        sa.Column("speed_i2", sa.Float()),
        sa.Column("speed_fl", sa.Float()),
        sa.Column("speed_st", sa.Float()),
        sa.Column("is_personal_best", sa.Boolean()),
        sa.Column("compound", sa.String(length=32)),
        sa.Column("tyre_life", sa.Float()),
        sa.Column("fresh_tyre", sa.Boolean()),
        sa.Column("stint", sa.Float()),
        sa.Column("track_status", sa.String(length=128)),
        sa.Column("position", sa.Float()),
        sa.Column("is_deleted", sa.Boolean()),
        sa.Column("deleted_reason", sa.Text()),
        sa.Column("is_fastf1_generated", sa.Boolean()),
        sa.Column("is_accurate", sa.Boolean()),
        sa.PrimaryKeyConstraint("session_id", "driver_number", "lap_number"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index("idx_session_laps", "session_laps", ["session_id"], unique=False)

    op.create_table(
        "circuit_features",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("feature_type", sa.String(length=32)),
        sa.Column("number", sa.Integer()),
        sa.Column("letter", sa.String(length=32)),
        sa.Column("circuit_rotation", sa.Float()),
        sa.Column("coordinate_x", sa.Float()),
        sa.Column("coordinate_y", sa.Float()),
        sa.Column("angle", sa.Float()),
        sa.Column("distance", sa.Float()),
        sa.PrimaryKeyConstraint("session_id", "feature_type", "number", "letter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_circuit_features", "circuit_features", ["session_id"], unique=False
    )

    op.create_table(
        "telemetry_position",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("driver_number", sa.String(length=32)),
        sa.Column("counter", sa.Integer()),
        sa.Column("timestamp_relative", sa.Float()),
        sa.Column("timestamp_absolute", sa.DateTime(timezone=True)),
        sa.Column("timestamp_alternative", sa.Float()),
        sa.Column("source", sa.String(length=128)),
        sa.Column("status", sa.String(length=128)),
        sa.Column("coordinate_x", sa.Float()),
        sa.Column("coordinate_y", sa.Float()),
        sa.Column("coordinate_z", sa.Float()),
        sa.PrimaryKeyConstraint("session_id", "driver_number", "counter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index(
        "idx_telemetry_position", "telemetry_position", ["session_id"], unique=False
    )

    op.create_table(
        "telemetry_car",
        sa.Column("session_id", sa.String(length=32)),
        sa.Column("driver_number", sa.String(length=32)),
        sa.Column("counter", sa.Integer()),
        sa.Column("timestamp_relative", sa.Float()),
        sa.Column("timestamp_absolute", sa.DateTime(timezone=True)),
        sa.Column("timestamp_alternative", sa.Float()),
        sa.Column("source", sa.String(length=128)),
        sa.Column("rpm", sa.Float()),
        sa.Column("speed", sa.Float()),
        sa.Column("gear", sa.Float()),
        sa.Column("throttle", sa.Float()),
        sa.Column("brake", sa.Float()),
        sa.Column("drs", sa.Float()),
        sa.PrimaryKeyConstraint("session_id", "driver_number", "counter"),
        sa.ForeignKeyConstraint(["session_id"], ["session_information.session_id"]),
    )
    op.create_index("idx_telemetry_car", "telemetry_car", ["session_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("session_messages")
    op.drop_table("session_status")
    op.drop_table("session_track_status")
    op.drop_table("session_results")
    op.drop_table("session_weather")
    op.drop_table("session_laps")
    op.drop_table("circuit_features")
    op.drop_table("telemetry_position")
    op.drop_table("telemetry_car")
    op.drop_table("session_information")
