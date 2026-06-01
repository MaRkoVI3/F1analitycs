import json
import os
import pandas as pd
from datetime import datetime

def load_json_file(filepath):
    """Load and return data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_race_results(data):
    """Extract race results from the MRData structure."""
    if not data or 'MRData' not in data:
        return []

    races = data['MRData'].get('RaceTable', {}).get('Races', [])
    results = []

    for race in races:
        race_info = {
            'season': race.get('season'),
            'round': race.get('round'),
            'race_name': race.get('raceName'),
            'date': race.get('date'),
            'time': race.get('time'),
            'circuit_id': race.get('Circuit', {}).get('circuitId'),
            'circuit_name': race.get('Circuit', {}).get('circuitName'),
            'country': race.get('Circuit', {}).get('Location', {}).get('country'),
            'locality': race.get('Circuit', {}).get('Location', {}).get('locality')
        }

        # Extract results for this race
        race_results = race.get('Results', [])
        for result in race_results:
            driver_info = result.get('Driver', {})
            constructor_info = result.get('Constructor', {})

            # Handle Time objects which might be strings or dicts
            time_obj = result.get('Time')
            time_value = None
            if time_obj:
                if isinstance(time_obj, dict):
                    time_value = time_obj.get('time')
                else:
                    time_value = str(time_obj)  # It's already a string like "1:30.499"

            # Handle FastestLap objects
            fastest_lap_obj = result.get('FastestLap')
            fastest_lap_time = None
            fastest_lap_speed = None
            if fastest_lap_obj:
                if isinstance(fastest_lap_obj, dict):
                    time_obj = fastest_lap_obj.get('Time')
                    if time_obj:
                        if isinstance(time_obj, dict):
                            fastest_lap_time = time_obj.get('time')
                        else:
                            fastest_lap_time = str(time_obj)

                    speed_obj = fastest_lap_obj.get('AverageSpeed')
                    if speed_obj:
                        if isinstance(speed_obj, dict):
                            fastest_lap_speed = speed_obj.get('speed')
                        else:
                            fastest_lap_speed = str(speed_obj)
                else:
                    # It's a string value
                    fastest_lap_time = str(fastest_lap_obj)

            result_record = {
                **race_info,
                'driver_id': driver_info.get('driverId'),
                'driver_number': driver_info.get('permanentNumber', ''),
                'driver_code': driver_info.get('code', ''),
                'driver_first_name': driver_info.get('givenName'),
                'driver_last_name': driver_info.get('familyName'),
                'driver_name': f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                'constructor_id': constructor_info.get('constructorId'),
                'constructor_name': constructor_info.get('name'),
                'grid_position': int(result.get('grid', 0)) if result.get('grid', '').isdigit() else 0,
                'finish_position': int(result.get('position', 0)) if result.get('position', '').isdigit() else 0,
                'points': float(result.get('points', 0)),
                'status': result.get('status'),
                'time': time_value,
                'fastest_lap': fastest_lap_time,
                'fastest_lap_speed': fastest_lap_speed
            }
            results.append(result_record)

    return results

def extract_qualifying_results(data):
    """Extract qualifying results from the MRData structure."""
    if not data or 'MRData' not in data:
        return []

    races = data['MRData'].get('RaceTable', {}).get('Races', [])
    results = []

    for race in races:
        race_info = {
            'season': race.get('season'),
            'round': race.get('round'),
            'race_name': race.get('raceName'),
            'date': race.get('date'),
            'time': race.get('time'),
            'circuit_id': race.get('Circuit', {}).get('circuitId'),
            'circuit_name': race.get('Circuit', {}).get('circuitName'),
            'country': race.get('Circuit', {}).get('Location', {}).get('country'),
            'locality': race.get('Circuit', {}).get('Location', {}).get('locality')
        }

        # Extract qualifying results for this race
        qualifying_results = race.get('QualifyingResults', [])
        for quali in qualifying_results:
            driver_info = quali.get('Driver', {})
            constructor_info = quali.get('Constructor', {})

            quali_record = {
                **race_info,
                'driver_id': driver_info.get('driverId'),
                'driver_number': driver_info.get('permanentNumber', ''),
                'driver_code': driver_info.get('code', ''),
                'driver_first_name': driver_info.get('givenName'),
                'driver_last_name': driver_info.get('familyName'),
                'driver_name': f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                'constructor_id': constructor_info.get('constructorId'),
                'constructor_name': constructor_info.get('name'),
                'quali_position': int(quali.get('position', 0)) if quali.get('position', '').isdigit() else 0,
                'q1_time': quali.get('Q1'),
                'q2_time': quali.get('Q2'),
                'q3_time': quali.get('Q3')
            }
            results.append(quali_record)

    return results

def process_season_data(year):
    """Process both results and qualifying data for a given year."""
    # Load results data
    results_file = f"/home/bobo/Projects/f1_predictor/data/raw/{year}_results.json"
    results_data = load_json_file(results_file)
    results = extract_race_results(results_data) if results_data else []

    # Load qualifying data
    quali_file = f"/home/bobo/Projects/f1_predictor/data/raw/{year}_qualifying.json"
    quali_data = load_json_file(quali_file)
    qualifying = extract_qualifying_results(quali_data) if quali_data else []

    return results, qualifying

def main():
    """Main function to process all seasons from 2021-2025."""
    # Create processed directory
    processed_dir = "/home/bobo/Projects/f1_predictor/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    all_results = []
    all_qualifying = []

    # Process each year
    for year in range(2021, 2026):
        print(f"Processing {year} data...")
        results, qualifying = process_season_data(year)
        all_results.extend(results)
        all_qualifying.extend(qualifying)
        print(f"  - Found {len(results)} race results")
        print(f"  - Found {len(qualifying)} qualifying results")

    # Convert to DataFrames
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = os.path.join(processed_dir, "race_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Saved race results to {results_csv}")

        # Also save summary statistics
        summary = {
            'total_races': results_df['race_name'].nunique(),
            'total_drivers': results_df['driver_id'].nunique(),
            'seasons': sorted(results_df['season'].unique().tolist()),
            'date_range': {
                'start': results_df['date'].min(),
                'end': results_df['date'].max()
            }
        }
        summary_file = os.path.join(processed_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")

    if all_qualifying:
        quali_df = pd.DataFrame(all_qualifying)
        quali_csv = os.path.join(processed_dir, "qualifying_results.csv")
        quali_df.to_csv(quali_csv, index=False)
        print(f"Saved qualifying results to {quali_csv}")

    # Create a combined dataset for ML
    if all_results and all_qualifying:
        # Merge results and qualifying on season, round, and driver_id
        results_df = pd.DataFrame(all_results)
        quali_df = pd.DataFrame(all_qualifying)

        # Merge on common keys
        merged_df = pd.merge(
            results_df,
            quali_df[['season', 'round', 'driver_id', 'quali_position', 'q1_time', 'q2_time', 'q3_time']],
            on=['season', 'round', 'driver_id'],
            how='left'
        )

        # Save combined dataset
        combined_csv = os.path.join(processed_dir, "combined_dataset.csv")
        merged_df.to_csv(combined_csv, index=False)
        print(f"Saved combined dataset to {combined_csv}")

        # Print some basic info
        print(f"\nDataset Info:")
        print(f"  - Total records: {len(merged_df)}")
        print(f"  - Features: {list(merged_df.columns)}")
        print(f"  - Missing quali data: {merged_df['quali_position'].isna().sum()} records")

if __name__ == "__main__":
    main()