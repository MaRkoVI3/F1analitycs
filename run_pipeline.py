#!/usr/bin/env python3
"""
F1 Predictor Pipeline Orchestrator
Glue script to run the complete F1 prediction pipeline:
1. Data Ingestion (download F1 data)
2. Feature Engineering (create ML features)
3. Simulation Engine (train model & run Monte Carlo predictions)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

def print_step(step_num, total_steps, description):
    """Print a formatted step indicator."""
    print(f"\nStep {step_num}/{total_steps}: {description}")
    print("-" * 50)

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"✓ Found {description}: {filepath}")
        return True
    else:
        print(f"✗ Missing {description}: {filepath}")
        return False

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"Running {description}...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ {description} completed successfully in {elapsed_time:.1f}s")
            return True
        else:
            print(f"✗ {description} failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False

def main():
    """Main pipeline orchestrator."""
    print_header("F1 PREDICTOR PIPELINE")
    print("This script will run the complete F1 prediction workflow:")
    print("1. Download F1 data (historical 2021-2025 + current 2026)")
    print("2. Engineer features for machine learning")
    print("3. Train model and run Monte Carlo simulations")

    # Initialize timing
    pipeline_start = time.time()

    # Define file paths
    data_dir = Path("./data")
    historical_data = data_dir / "historical_results_2021_2025.csv"
    featured_data = data_dir / "driver_features.csv"

    print_step(1, 3, "Checking Environment")
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    # Check if we can skip data ingestion
    skip_ingestion = historical_data.exists()
    if skip_ingestion:
        print(f"✓ Historical data already exists: {historical_data}")
        user_input = input("Use existing data? (y/n, default=y): ").strip().lower()
        if user_input == 'n':
            skip_ingestion = False
            print("Will re-download data...")
        else:
            print("Using existing data...")
    else:
        print("✗ Historical data not found - will download")

    # Step 1: Data Ingestion
    print_step(1, 3, "Data Ingestion - Downloading F1 Data")
    if not skip_ingestion:
        success = run_script("data_ingestion.py", "Data Ingestion")
        if not success:
            print("\n✗ Pipeline failed at data ingestion step.")
            print("Please check your internet connection and try again.")
            return 1
    else:
        print("✓ Skipping data ingestion (using existing data)")

    # Verify historical data exists after ingestion (or skipping)
    if not check_file_exists(str(historical_data), "historical data file"):
        print("\n✗ Cannot proceed without historical data.")
        return 1

    # Step 2: Feature Engineering
    print_step(2, 3, "Feature Engineering - Creating ML Features")
    success = run_script("feature_engineering.py", "Feature Engineering")
    if not success:
        print("\n✗ Pipeline failed at feature engineering step.")
        return 1

    # Verify featured data exists
    if not check_file_exists(str(featured_data), "featured data file"):
        print("\n✗ Cannot proceed without featured data.")
        return 1

    # Step 3: Simulation Engine
    print_step(3, 3, "Simulation Engine - Training Model & Running Predictions")
    success = run_script("simulation_engine.py", "Simulation Engine")
    if not success:
        print("\n✗ Pipeline failed at simulation step.")
        return 1

    # Pipeline completed successfully
    total_time = time.time() - pipeline_start
    print_header("PIPELINE COMPLETED SUCCESSFULLY 🏁")
    print(f"Total execution time: {total_time:.1f} seconds")
    print("\nOutput files generated:")
    print(f"  - Raw data: {data_dir}/")
    print(f"  - Features: {featured_data}")
    print(f"  - Model: {data_dir}/ (if saved by simulation_engine)")
    print(f"  - Predictions: Check console output above")

    print("\nNext steps:")
    print("- Examine the prediction probabilities shown above")
    print("- To re-run with fresh data, delete the ./data/ directory")
    print("- To modify parameters, edit the individual scripts")

    return 0

if __name__ == "__main__":
    sys.exit(main())