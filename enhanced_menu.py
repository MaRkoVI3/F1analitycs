#!/usr/bin/env python3
"""
Unified F1 Predictor Menu System
"""

import subprocess
import sys
import os
import json
import glob
from datetime import datetime

def print_header():
    print("\n" + "="*60)
    print("F1 PREDICTOR ENHANCED MENU")
    print("="*60)
    print("Predicting F1 race outcomes using ML & Monte Carlo simulation")
    print("Training data: 2021-2025 | Target: 2026 season predictions")
    print("="*60)

def print_menu():
    print("\nMAIN MENU:")
    print("1. Update F1 Data (Fetch 2021-2025 from API)")
    print("2. Process Raw Data → Features")
    print("3. Train ML Model (Random Forest)")
    print("4. Predict 2026 Season Races")
    print("5. View Prediction Results & Statistics")
    print("6. Driver/Team Track Analysis")
    print("7. Championship Standings Viewer")
    print("8. Historical Track Performance")
    print("9. Monte Carlo Race Simulation")
    print("10. Full Pipeline (1→2→3→4)")
    print("0. Exit")
    print("-" * 60)

def run_script(script_name, description, show_output=True):
    """Run a Python script and optionally show output."""
    if show_output:
        print(f"\n{description}...")
        print("-" * 50)

    try:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        result = subprocess.run([sys.executable, script_path],
                              capture_output=True, text=True, timeout=120)

        if show_output:
            print(result.stdout)
            if result.stderr:
                print("  STDERR:", result.stderr[:200] + ("..." if len(result.stderr) > 200 else ""))
            print(f"  {description} completed.")
        else:
            # Silent run
            if result.returncode != 0:
                print(f"  {description} failed: {result.stderr[:100]}")
                return False

        if result.returncode != 0 and show_output:
            print(f"  Warning: Script exited with code {result.returncode}")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"  {description} timed out (2 min limit)")
        return False
    except Exception as e:
        print(f"  Error running {script_name}: {e}")
        return False

def view_results():
    """View the latest prediction results."""
    print("\n" + "="*50)
    print("VIEWING PREDICTION RESULTS")
    print("="*50)
    predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
    if not os.path.exists(predictions_dir):
        print("No predictions directory found. Run predictions first.")
        return

    # List available prediction files
    prediction_files = glob.glob(os.path.join(predictions_dir, '*.csv'))
    if not prediction_files:
        print("No prediction CSV files found.")
        return

    print("Available prediction files:")
    for i, f in enumerate(prediction_files, 1):
        print(f"  {i}. {os.path.basename(f)}")

    try:
        idx = int(input("\nSelect a file to view (number): ")) - 1
        if 0 <= idx < len(prediction_files):
            selected_file = prediction_files[idx]
            print(f"\nContents of {os.path.basename(selected_file)}:")
            print("-" * 50)
            # Show first 25 lines
            with open(selected_file, 'r') as f:
                lines = f.readlines()
                for line in lines[:25]:
                    print(line.rstrip())
                if len(lines) > 25:
                    print(f"... and {len(lines) - 25} more lines")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")

def run_full_pipeline():
    """Run the complete data fetch → process → train → predict pipeline."""
    print("\n" + "="*50)
    print("RUNNING FULL PIPELINE")
    print("="*50)
    steps = [
        ('src/fetch_data.py', 'Fetching F1 data (2021-2025)'),
        ('src/process_data.py', 'Processing raw data → features'),
        ('src/train_model.py', 'Training ML model (Random Forest)'),
        ('src/predict_season.py', 'Generating 2026 season predictions')
    ]

    for script, description in steps:
        print(f"\nStep: {description}")
        if not run_script(script, description, show_output=False):
            print(f"Pipeline failed at step: {description}")
            return

    print("\n" + "="*50)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)

def driver_team_track_analysis():
    """Analyze driver/team performance at a specific track."""
    print("\n" + "="*50)
    print("DRIVER/TEAM TRACK ANALYSIS")
    print("="*50)

    # Load processed data
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed', 'combined_dataset.csv')
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} race records (2021-2025)")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Get unique circuits
    circuits = df[['circuit_id', 'circuit_name', 'country', 'locality']].drop_duplicates()
    print("\n🏁 Available circuits:")
    for i, (_, row) in enumerate(circuits.iterrows(), 1):
        print(f"  {i:2d}. {row['circuit_name']} ({row['locality']}, {row['country']})")

    try:
        choice = int(input("\n🔢 Select circuit number: ")) - 1
        if 0 <= choice < len(circuits):
            selected_circuit = circuits.iloc[choice]
            circuit_id = selected_circuit['circuit_id']
            circuit_name = selected_circuit['circuit_name']

            print(f"\n📈 Analyzing performance at: {circuit_name}")
            print("-" * 50)

            # Filter data for this circuit
            circuit_data = df[df['circuit_id'] == circuit_id].copy()

            if len(circuit_data) == 0:
                print("📭 No data available for this circuit.")
                return

            # Driver statistics at this track
            driver_stats = circuit_data.groupby(['driver_id', 'driver_name', 'constructor_name']).agg({
                'finish_position': ['mean', 'min', 'max', 'count', 'std'],
                'points': 'mean',
                'grid_position': 'mean'
            }).round(2)

            # Flatten column names
            driver_stats.columns = ['avg_finish', 'best_finish', 'worst_finish', 'races_count',
                                  'finish_std', 'avg_points', 'avg_grid']
            driver_stats = driver_stats.reset_index()

            # Sort by average finish (best first)
            driver_stats = driver_stats.sort_values('avg_finish')

            print(f"Driver Performance at {circuit_name}:")
            print(f"{'Driver':<20} {'Team':<15} {'Avg':<5} {'Best':<5} {'Worst':<5} {'Races':<6} {'Pts':<5}")
            print("-" * 80)
            for _, row in driver_stats.head(10).iterrows():
                print(f"{row['driver_name']:<20} {row['constructor_name']:<15} "
                      f"{row['avg_finish']:<5.1f} {row['best_finish']:<5.0f} "
                      f"{row['worst_finish']:<5.0f} {row['races_count']:<6.0f} "
                      f"{row['avg_points']:<5.1f}")

            # Team statistics at this track
            team_stats = circuit_data.groupby('constructor_name').agg({
                'finish_position': ['mean', 'min', 'count'],
                'points': 'mean'
            }).round(2)
            team_stats.columns = ['avg_finish', 'best_finish', 'races_count', 'avg_points']
            team_stats = team_stats.reset_index().sort_values('avg_finish')

            print(f"\n🏭 Team Performance at {circuit_name}:")
            print(f"{'Team':<15} {'Avg':<5} {'Best':<5} {'Races':<6} {'Pts':<5}")
            print("-" * 50)
            for _, row in team_stats.head(8).iterrows():
                print(f"{row['constructor_name']:<15} {row['avg_finish']:<5.1f} "
                      f"{row['best_finish']:<5.0f} {row['races_count']:<6.0f} "
                      f"{row['avg_points']:<5.1f}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"Error in analysis: {e}")

def championship_standings_viewer():
    """View championship standings and predictions."""
    print("\n" + "="*50)
    print("CHAMPIONSHIP STANDINGS VIEWER")
    print("="*50)

    try:
        # Load predictions if available
        predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')

        # Look for latest predictions
        pred_files = glob.glob(os.path.join(predictions_dir, '*_race_predictions.csv'))
        if pred_files:
            latest_pred = max(pred_files, key=os.path.getctime)
            import pandas as pd
            predictions_df = pd.read_csv(latest_pred)
            season = os.path.basename(latest_pred).split('_')[0]
            print(f"📅 Showing predictions for {season} season:")
            print("-" * 50)

            # Calculate championship from predictions
            champ = predictions_df.groupby(['driver_name', 'constructor_name'])['predicted_points'].sum().reset_index()
            champ = champ.rename(columns={'predicted_points': 'total_points'})
            champ = champ.sort_values('total_points', ascending=False)
            champ['position'] = range(1, len(champ) + 1)

            print("🥇 PREDICTED CHAMPIONSHIP:")
            print(f"{'Pos':<4} {'Driver':<20} {'Team':<15} {'Points':<8}")
            print("-" * 50)
            for _, row in champ.head(10).iterrows():
                print(f"{row['position']:<4} {row['driver_name']:<20} {row['constructor_name']:<15} {row['total_points']:<8.0f}")

        # Load actual results if available
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed', 'race_results.csv')
        if os.path.exists(data_path):
            results_df = pd.read_csv(data_path)
            # Get latest season available
            latest_season = results_df['season'].max()
            season_data = results_df[results_df['season'] == latest_season]

            if len(season_data) > 0:
                actual_champ = season_data.groupby(['driver_name', 'constructor_name'])['points'].sum().reset_index()
                actual_champ = actual_champ.sort_values('points', ascending=False)
                actual_champ['position'] = range(1, len(actual_champ) + 1)

                print(f"\nACTUAL CHAMPIONSHIP ({latest_season}):")
                print(f"{'Pos':<4} {'Driver':<20} {'Team':<15} {'Points':<8}")
                print("-" * 50)
                for _, row in actual_champ.head(10).iterrows():
                    print(f"{row['position']:<4} {row['driver_name']:<20} {row['constructor_name']:<15} {row['points']:<8.0f}")

                # Compare if we have both
                if 'predictions_df' in locals():
                    print(f"\n🔍 COMPARISON (Predicted vs Actual {latest_season}):")
                    # Merge for comparison
                    merged = pd.merge(
                        champ[['driver_name', 'total_points']].rename(columns={'total_points': 'predicted'}),
                        actual_champ[['driver_name', 'points']].rename(columns={'points': 'actual'}),
                        on='driver_name', how='inner'
                    )
                    if len(merged) > 0:
                        merged['diff'] = merged['predicted'] - merged['actual']
                        merged['abs_diff'] = abs(merged['diff'])
                        merged = merged.sort_values('abs_diff')

                        print(f"{'Driver':<20} {'Pred':<6} {'Actual':<6} {'Diff':<6}")
                        print("-" * 40)
                        for _, row in merged.head(8).iterrows():
                            print(f"{row['driver_name']:<20} {row['predicted']:<6.0f} {row['actual']:<6.0f} {row['diff']:<6.0f}")
        else:
            print("📭 No actual results data found. Run data processing first.")

    except Exception as e:
        print(f"Error loading championship data: {e}")

def historical_track_performance():
    """Show historical performance at a specific track."""
    print("\n" + "="*50)
    print("📈 HISTORICAL TRACK PERFORMANCE")
    print("="*50)

    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed', 'combined_dataset.csv')
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} race records (2021-2025)")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Get unique circuits
    circuits = df[['circuit_id', 'circuit_name', 'country', 'locality']].drop_duplicates()
    print("\n🏁 Available circuits:")
    for i, (_, row) in enumerate(circuits.iterrows(), 1):
        print(f"  {i:2d}. {row['circuit_name']} ({row['locality']}, {row['country']})")

    try:
        choice = int(input("\n🔢 Select circuit number: ")) - 1
        if 0 <= choice < len(circuits):
            selected_circuit = circuits.iloc[choice]
            circuit_id = selected_circuit['circuit_id']
            circuit_name = selected_circuit['circuit_name']

            print(f"\n📈 Historical performance at {circuit_name}:")
            print("-" * 50)

            # Filter data for this circuit
            circuit_data = df[df['circuit_id'] == circuit_id].copy()

            if len(circuit_data) == 0:
                print("📭 No data available for this circuit.")
                return

            # Year-by-year performance
            yearly = circuit_data.groupby(['season', 'race_name']).agg({
                'driver_name': 'count',  # Number of finishers
                'points': 'mean',        # Average points scored
                'finish_position': 'mean', # Average finish position
                'status': lambda x: (x == 'Finished').sum()  # Finishers
            }).reset_index()

            yearly.columns = ['season', 'race_name', 'finishers', 'avg_points', 'avg_position', 'finished_count']
            yearly = yearly.sort_values('season')

            print(f"{'Season':<6} {'Race':<25} {'Finishers':<10} {'Avg Points':<10} {'Avg Pos':<8}")
            print("-" * 70)
            for _, row in yearly.iterrows():
                print(f"{row['season']:<6} {row['race_name']:<25} {row['finishers']:<10.0f} "
                      f"{row['avg_points']:<10.1f} {row['avg_position']:<8.1f}")

            # Most successful drivers at this track
            print(f"\nMost Successful Drivers at {circuit_name}:")
            driver_success = circuit_data.groupby(['driver_name', 'constructor_name']).agg({
                'points': 'sum',
                'finish_position': 'mean',
                'race_name': 'count'
            }).reset_index()
            driver_success.columns = ['driver', 'team', 'total_points', 'avg_position', 'races']
            driver_success = driver_success.sort_values('total_points', ascending=False)

            print(f"{'Driver':<20} {'Team':<15} {'Points':<8} {'Avg Pos':<8} {'Races':<6}")
            print("-" * 60)
            for _, row in driver_success.head(8).iterrows():
                print(f"{row['driver']:<20} {row['team']:<15} {row['total_points']:<8.0f} "
                      f"{row['avg_position']:<8.1f} {row['races']:<6.0f}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"Error in historical analysis: {e}")

def monte_carlo_simulation():
    """Run Monte Carlo race simulation."""
    print("\n" + "="*50)
    print("MONTE CARLO RACE SIMULATION")
    print("="*50)

    print("This would run the interactive simulation from F1analitycs/")
    print("For now, running the existing simulation engine...")

    # Change to F1analitycs directory and run simulation
    f1analitycs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'F1analitycs')
    if os.path.exists(f1analitycs_path):
        original_dir = os.getcwd()
        try:
            os.chdir(f1analitycs_path)
            print("\nLaunching Monte Carlo simulation...")
            result = subprocess.run([sys.executable, 'simulation_engine.py'],
                                  capture_output=True, text=True, timeout=60)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            os.chdir(original_dir)
        except Exception as e:
            print(f"Error running simulation: {e}")
            os.chdir(original_dir)
    else:
        print("F1analitycs directory not found.")

def main():
    """Main menu loop."""
    print_header()

    while True:
        print_menu()
        try:
            choice = input("\nEnter your choice (0-10): ").strip()
        except EOFError:
            # Handle non-interactive environment
            print("\nNo input available. Showing menu help.")
            print("Run with input available for interactive use.")
            break

        if choice == '0':
            print("\nExiting F1 Predictor. Goodbye!")
            break
        elif choice == '1':
            run_script('src/fetch_data.py', 'Fetching F1 data from API (2021-2025)')
        elif choice == '2':
            run_script('src/process_data.py', 'Processing raw data into features')
        elif choice == '3':
            run_script('src/train_model.py', 'Training ML model (Random Forest)')
        elif choice == '4':
            run_script('src/predict_season.py', 'Generating 2026 season predictions')
        elif choice == '5':
            view_results()
        elif choice == '6':
            driver_team_track_analysis()
        elif choice == '7':
            championship_standings_viewer()
        elif choice == '8':
            historical_track_performance()
        elif choice == '9':
            monte_carlo_simulation()
        elif choice == '10':
            run_full_pipeline()
        else:
            print("Invalid choice. Please enter a number between 0 and 10.")

if __name__ == "__main__":
    main()