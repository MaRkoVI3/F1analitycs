import os
import sys
import logging
import fastf1
import pandas as pd

# 1. Setup Logging for Production-ready visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 2. Configure Local Directory and FastF1 Cache
DATA_DIR = './data'
CACHE_DIR = './f1_cache'

for directory in [DATA_DIR, CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

# CRUCIAL: Enable local caching to prevent API rate limits and speed up runs
fastf1.Cache.enable_cache(CACHE_DIR)
logger.info("FastF1 cache successfully enabled locally.")

def download_historical_data(start_year=2021, end_year=2025):
    """
    Downloads historical race results to construct the Driver Incident Index 
    and Track History baselines.
    """
    logger.info(f"Starting historical data download from {start_year} to {end_year}...")
    historical_records = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            logger.error(f"Failed to fetch schedule for year {year}: {e}")
            continue

        for _, row in schedule.iterrows():
            round_num = row['RoundNumber']
            # Round 0 represents pre-season testing, skip it
            if round_num == 0:
                continue
            
            event_name = row['EventName']
            logger.info(f"Fetching {year} - Round {round_num}: {event_name}")

            try:
                # Load Race session ('R') without heavy telemetry to save space/time for history
                session = fastf1.get_session(year, round_num, 'R')
                session.load(telemetry=False, weather=False, messages=False)
                
                results_df = session.results
                if results_df.empty:
                    continue

                # Extract features critical for our upcoming local ML model
                extracted_data = pd.DataFrame({
                    'Year': year,
                    'RoundNumber': round_num,
                    'Track': event_name,
                    'DriverNumber': results_df['DriverNumber'],
                    'DriverAbbreviation': results_df['Abbreviation'],
                    'TeamName': results_df['TeamName'],
                    'GridPosition': results_df['GridPosition'],
                    'FinalPosition': results_df['Position'],
                    'Status': results_df['Status'],  # Crucial for Incident Index (e.g., 'Collision')
                    'Points': results_df['Points']
                })
                
                historical_records.append(extracted_data)

            except Exception as e:
                logger.warning(f"Skipping {year} Round {round_num} due to error: {e}")

    if historical_records:
        master_history_df = pd.concat(historical_records, ignore_index=True)
        output_path = os.path.join(DATA_DIR, 'historical_results_2021_2025.csv')
        master_history_df.to_csv(output_path, index=False)
        logger.info(f"Master historical dataset saved to {output_path} ({len(master_history_df)} rows).")
    else:
        logger.error("No historical data was collected.")

def download_current_season_data(year=2026):
    """
    Downloads high-resolution telemetry, lap logs, and track safety statuses 
    for completed races in the current 2026 season.
    """
    logger.info(f"Starting telemetry data download for the current {year} season...")
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        logger.critical(f"Could not load 2026 schedule: {e}")
        return

    for _, row in schedule.iterrows():
        round_num = row['RoundNumber']
        if round_num == 0:
            continue
            
        event_name = row['EventName']
        
        try:
            session = fastf1.get_session(year, round_num, 'R')
            # Load full telemetry, lap, and weather streams for risk profile maps
            session.load(telemetry=True, laps=True, weather=True)
            
            # 1. Export Lap times and micro-sectors
            laps_df = session.laps
            laps_path = os.path.join(DATA_DIR, f'{year}_round_{round_num}_laps.csv')
            laps_df.to_csv(laps_path, index=False)
            
            # 2. Export Track Safety Status (Green, Yellow, Safety Car, Red Flag)
            # This is the backbone of your Track Volatility calculation!
            status_df = session.track_status
            status_path = os.path.join(DATA_DIR, f'{year}_round_{round_num}_status.csv')
            status_df.to_csv(status_path, index=False)
            
            logger.info(f"Successfully downloaded and saved deep telemetry for 2026 Round {round_num} ({event_name}).")
            
        except Exception as e:
            # If a race hasn't happened yet in 2026, FastF1 will throw an error. We handle it gracefully.
            logger.info(f"Round {round_num} ({event_name}) could not be loaded. It may not have occurred yet.")

if __name__ == "__main__":
    logger.info("Initializing Data Ingestion Engine...")
    
    # Step A: Build historical feature bank
    download_historical_data()
    
    # Step B: Build current season telemetry telemetry bank
    download_current_season_data()
    
    logger.info("Day 1 Data Ingestion completed successfully!")