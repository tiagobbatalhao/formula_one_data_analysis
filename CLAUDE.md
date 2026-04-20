# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Formula 1 data analysis project using the fastf1 library. The project implements a medallion architecture (bronze/silver/gold layers) for data processing and includes a Streamlit-based visualization application.

## Architecture

### Data Layers (Medallion Architecture)

The project organizes data processing into three layers defined in `src/data_engineering/`:

- **Bronze Layer** (`bronze_layer.py`): Raw data from fastf1 API, saved as Parquet files with minimal transformation. Includes session metadata, results, laps, weather, telemetry, and circuit data.
- **Silver Layer** (`silver_layer.py`): Cleaned and standardized data, combining multiple bronze sources.
- **Gold Layer** (`gold_layer.py`): Aggregated, business-ready datasets optimized for visualization and analysis.

All layers inherit from `DatasetLocal` class in `datasets.py`, which handles:
- Reading/saving Parquet files to `data/` directory
- Force refresh via `force=True` parameter
- Pattern-based loading for multiple files

### Data Flow

1. **Download**: `download_event.py` or `download_history.py` fetch data from fastf1 → writes to bronze layer
2. **Process**: `update_downstream_layers.py` transforms bronze → silver → gold
3. **Visualize**: Streamlit app reads from gold layer

### Session Identification

Sessions use a three-part identifier: `(year, round_id, session_number)`
- `round_id` format: `R{number}` for official races (e.g., "R03"), `T{number}` for testing
- `session_number`: 1-5 for races (FP1, FP2, FP3, Qualifying, Race), 1-3 for testing

## Common Commands

### Data Download

Download a specific event (all sessions):
```bash
python src/download_event.py <year> <round_id>
```

Download a specific session:
```bash
python src/download_event.py <year> <round_id> --session_id <session_number>
```

Examples:
```bash
python src/download_event.py 2026 R03              # Download all sessions for Round 3
python src/download_event.py 2026 R03 --session_id 5  # Download only Race session
python src/download_event.py 2026 T01              # Download testing event
```

Force re-download (ignore cache):
```bash
python src/download_event.py <year> <round_id> --force
```

Download historical race schedule:
```bash
python src/download_history.py --year_start 1950 --year_end 2026
```

### Data Processing

Update silver and gold layers for a specific year:
```bash
python src/update_downstream_layers.py <year>
```

### Visualization

Run the Streamlit visualization app:
```bash
streamlit run src/visualization_app/streamlit_app.py
```

The app must be run from the project root directory so it can access the `data/` folder.

### Circuit Map Modeling

Run circuit map modeling with MLflow tracking:
```bash
python src/run_circuit_map.py --year <year> --round_id <round_id> --session_id <session_id>
```

Plot circuit map:
```bash
python src/plot_circuit_map.py --year <year> --round_id <round_id> --session_id <session_id>
```

## Project Structure

```
.
├── src/
│   ├── data_engineering/        # Data layer definitions
│   │   ├── datasets.py          # Base DatasetLocal class
│   │   ├── bronze_layer.py      # Raw data from fastf1
│   │   ├── silver_layer.py      # Cleaned data
│   │   └── gold_layer.py        # Aggregated data
│   ├── visualization_app/       # Streamlit application
│   │   ├── streamlit_app.py     # Main app entry point
│   │   ├── data_loader.py       # Data loading utilities
│   │   ├── tab_*.py             # Individual tab implementations
│   │   └── circuit_map_utils.py # Circuit visualization utilities
│   ├── download_event.py        # Download specific event data
│   ├── download_history.py      # Download historical schedules
│   ├── update_downstream_layers.py  # Process data layers
│   ├── run_circuit_map.py       # Circuit modeling with MLflow
│   └── plot_circuit_map.py      # Circuit visualization
├── cache/                       # fastf1 cache (gitignored)
├── data/                        # Parquet data files (gitignored)
│   ├── bronze/
│   ├── silver/
│   └── gold/
└── mlruns/                      # MLflow tracking (gitignored)
```

## Key Technologies

- **fastf1**: Official F1 timing data API
- **pandas**: Data manipulation
- **streamlit**: Interactive visualization app
- **plotly**: Interactive plots
- **mlflow**: Experiment tracking for circuit map modeling
- **scikit-learn, scipy**: Circuit map fitting using Fourier series

## Important Notes

### fastf1 Cache

The fastf1 library caches downloaded data in `cache/`. This speeds up subsequent loads but may need force refresh if upstream data changes. The cache is enabled in `download_event.py`:

```python
cache_dir = Path(__file__).parent.parent / "cache"
fastf1.Cache.enable_cache(cache_dir=cache_dir)
```

### File Naming Conventions

Bronze layer files use pattern: `{entity}_Y{year}R{round:02d}S{session}.parquet`

Examples:
- `session_metadata_Y2026R03S1.parquet` - Practice 1 metadata for Round 3, 2026
- `telemetry_car_Y2026R03S5.parquet` - Race telemetry for Round 3, 2026

### MLflow

Circuit map modeling uses MLflow with SQLite backend (`mlflow.db`). Experiments are tracked under "formula_one_circuit_map". The modeling fits circuit coordinates using Fourier series to smooth GPS noise.

### Streamlit App Tabs

The visualization app (`streamlit_app.py`) includes:
- **Result**: Session classification/results
- **Lap history**: Lap-by-lap timing
- **Race trace**: Position changes over time
- **Lap telemetry**: Speed, throttle, brake, gear per lap
- **Circuit map**: Track layout visualization
