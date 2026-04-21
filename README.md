# Formula One Data Analysis

Data analysis with Formula 1 data provided by [fastf1](https://github.com/theOehrly/Fast-F1).

## Overview

This project implements a medallion architecture (bronze/silver/gold layers) for processing Formula 1 telemetry and timing data, with an interactive Streamlit visualization dashboard.

## Quick Start with Docker

### Prerequisites

- Docker
- Docker Compose (optional, but recommended)

### Build and Run

1. **Build the Docker image:**
   ```bash
   docker build -t formula_one_data_analysis .
   ```

2. **Download F1 data:**
   ```bash
   docker run -v $(pwd)/data:/app/data formula_one_data_analysis src/download_event.py 2026 R03
   ```

3. **Process the data:**
   ```bash
   docker run -v $(pwd)/data:/app/data formula_one_data_analysis src/update_downstream_layers.py 2026
   ```

4. **Launch the visualization app:**
   ```bash
   docker-compose up streamlit
   ```

   Or without Docker Compose:
   ```bash
   docker run -p 8501:8501 -v $(pwd)/data:/app/data formula_one_data_analysis streamlit run src/visualization_app/streamlit_app.py --server.address 0.0.0.0
   ```

5. **Open your browser:** http://localhost:8501

## Local Development Setup

### Prerequisites

- Python 3.12+
- pip

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download F1 data:**
   ```bash
   python src/download_event.py 2026 R03
   ```

3. **Process the data:**
   ```bash
   python src/update_downstream_layers.py 2026
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run src/visualization_app/streamlit_app.py
   ```

## Usage

### Download Data

Download all sessions for a specific event:
```bash
# Local
python src/download_event.py <year> <round_id>

# Docker
docker run -v $(pwd)/data:/app/data formula_one_data_analysis src/download_event.py <year> <round_id>
```

Where:
- `round_id`: Use `R01`, `R02`, etc. for race weekends, or `T01`, `T02` for testing

Download a specific session:
```bash
python src/download_event.py 2026 R03 --session_id 5  # 5 = Race
```

Sessions: 1=FP1, 2=FP2, 3=FP3, 4=Qualifying, 5=Race

### Process Data Layers

After downloading, process the data through silver and gold layers:
```bash
# Local
python src/update_downstream_layers.py <year>

# Docker
docker run -v $(pwd)/data:/app/data formula_one_data_analysis src/update_downstream_layers.py <year>
```

### Visualization Dashboard

The Streamlit app provides interactive visualizations:
- **Result**: Final classification and results
- **Lap history**: Lap-by-lap timing analysis
- **Race trace**: Position changes throughout the session
- **Lap telemetry**: Speed, throttle, brake, and gear data
- **Circuit map**: Track layout visualization

## Architecture

### Data Layers

- **Bronze**: Raw data from fastf1 API
- **Silver**: Cleaned and standardized data
- **Gold**: Aggregated, analysis-ready datasets

### Data Flow

1. fastf1 API → Bronze layer (Parquet files)
2. Bronze → Silver layer (cleaned data)
3. Silver → Gold layer (aggregated data)
4. Streamlit app reads from Gold layer

## Technologies

- **fastf1**: F1 timing data API
- **pandas**: Data manipulation
- **Streamlit**: Interactive dashboard
- **Plotly**: Interactive visualizations
- **MLflow**: Experiment tracking
- **Docker**: Containerization

## License

This project uses data from the fastf1 library. Please refer to [fastf1's documentation](https://theoehrly.github.io/Fast-F1/) for data usage terms.
