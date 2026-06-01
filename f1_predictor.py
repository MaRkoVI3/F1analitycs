#!/usr/bin/env python3
"""
F1 Predictor - Main Application
Command-line interface for the F1 prediction system.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def print_banner():
    banner = """
    ╔═════════════════════════════════════════════════════════════════════════════╗
    ║                        F1 PREDICTOR SYSTEM                        ║
    ║                                                                              ║
    ║  Predict Formula 1 race outcomes using:                                      ║
    ║  • Historical data analysis (2021-2025)                                      ║
    ║  • Driver Incident Intelligence & Track Volatility modeling                 ║
    ║  • Machine Learning + Monte Carlo simulation                                ║
    ║                                                                              ║
    ║  Components: data_ingestion.py → feature_engineering.py → simulation_engine.py║
    ╚═════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_step(step_num, total_steps, description):
    print(f"\nStep {step_num}/{total_steps}: {description}")
    print("-" * 60)

def run_script(script_name, description, show_output=True):
    print(f"Running {description}...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        elapsed_time = time.time() - start_time

        if show_output:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"{description} completed successfully in {elapsed_time:.1f}s")
            return True
        else:
            print(f"{description} failed with exit code {result.returncode}")
            if result.stderr and not show_output:
                print("Error output:", result.stderr)
            return False

    except Exception as e:
        print(f"Error running {description}: {e}")
        return False

def check_requirements():
    required_packages = ['fastf1', 'pandas', 'numpy', 'scikit-learn']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing required packages:", ", ".join(missing))
        print("Install them with: pip install -r requirements.txt")
        return False
    return True

def show_data_status():
    data_dir = Path("./data")
    historical_file = data_dir / "historical_results_2021_2025.csv"
    featured_file = data_dir / "driver_features.csv"

    print("\nData Status:")
    print("-" * 40)

    if historical_file.exists():
        size = historical_file.stat().st_size / 1024
        print(f"✓ Historical data: {historical_file.name} ({size:.1f} KB)")
    else:
        print("✗ Historical data: Not found (will be downloaded)")

    if featured_file.exists():
        size = featured_file.stat().st_size / 1024
        print(f"✓ Featured data: {featured_file.name} ({size:.1f} KB)")
    else:
        print("✗ Featured data: Not found (will be generated)")

def run_full_pipeline():
    print_banner()

    if not check_requirements():
        return 1

    show_data_status()

    print("\nReady to run the complete F1 prediction pipeline")
    print("This will:")
    print("1. Download historical F1 data (2021-2025) from FastF1")
    print("2. Download current season (2026) telemetry data")
    print("3. Engineer features (Incident Index, Recent Form, Track History)")
    print("4. Train a Machine Learning model (Random Forest)")
    print("5. Run Monte Carlo simulations (10,000 races) for predictions")

    confirm = input("\nContinue? (y/n, default=y): ").strip().lower()
    if confirm == 'n':
        print("Pipeline cancelled.")
        return 0

    pipeline_start = time.time()

    print_step(1, 3, "Data Ingestion - Downloading F1 Data")
    if not run_script("data_ingestion.py", "Data Ingestion"):
        print("\nPipeline failed at data ingestion step.")
        print("Check your internet connection and try again.")
        return 1

    print_step(2, 3, "Feature Engineering - Creating ML Features")
    if not run_script("feature_engineering.py", "Feature Engineering"):
        print("\nPipeline failed at feature engineering step.")
        return 1

    print_step(3, 3, "Simulation Engine - Training Model & Running Predictions")
    if not run_script("simulation_engine.py", "Simulation Engine"):
        print("\nPipeline failed at simulation step.")
        return 1

    total_time = time.time() - pipeline_start
    print_banner()
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {total_time:.1f} seconds")
    print("\nGenerated files:")
    print("  - ./data/ - Contains all downloaded and processed data")
    print("  - Check console output above for prediction probabilities")
    print("\nTips:")
    print("  - To re-run with fresh data, delete the ./data/ directory")
    print("  - To see detailed logs, check the output above")
    print("  - The simulation uses 10,000 Monte Carlo runs for accuracy")

    return 0

def run_data_only():
    print_banner()
    print("DOWNLOADING F1 DATA ONLY")
    print("-" * 40)

    if not check_requirements():
        return 1

    show_data_status()
    confirm = input("\nDownload/update F1 data? (y/n, default=y): ").strip().lower()
    if confirm == 'n':
        print("Data download cancelled.")
        return 0

    start_time = time.time()
    if run_script("data_ingestion.py", "Data Ingestion"):
        elapsed = time.time() - start_time
        print(f"\nData download completed in {elapsed:.1f}s")
        show_data_status()
        return 0
    else:
        print("\nData download failed.")
        return 1

def run_features_only():
    print_banner()
    print("FEATURE ENGINEERING ONLY")
    print("-" * 40)

    data_dir = Path("./data")
    historical_file = data_dir / "historical_results_2021_2025.csv"

    if not historical_file.exists():
        print("Historical data not found. Please run data ingestion first.")
        print("Use: python f1_predictor.py --data")
        return 1

    show_data_status()
    confirm = input("\nEngineer features from existing data? (y/n, default=y): ").strip().lower()
    if confirm == 'n':
        print("Feature engineering cancelled.")
        return 0

    start_time = time.time()
    if run_script("feature_engineering.py", "Feature Engineering"):
        elapsed = time.time() - start_time
        print(f"\nFeature engineering completed in {elapsed:.1f}s")
        show_data_status()
        return 0
    else:
        print("\nFeature engineering failed.")
        return 1

def run_simulation_only():
    print_banner()
    print("SIMULATION ONLY")
    print("-" * 40)

    data_dir = Path("./data")
    featured_file = data_dir / "driver_features.csv"

    if not featured_file.exists():
        print("Featured data not found. Please run feature engineering first.")
        print("Use: python f1_predictor.py --features")
        return 1

    show_data_status()
    confirm = input("\nRun simulation from existing features? (y/n, default=y): ").strip().lower()
    if confirm == 'n':
        print("Simulation cancelled.")
        return 0

    start_time = time.time()
    if run_script("simulation_engine.py", "Simulation Engine"):
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.1f}s")
        print("\nCheck the output above for prediction probabilities")
        return 0
    else:
        print("\nSimulation failed.")
        return 1

def show_help():
    print_banner()
    print("USAGE:")
    print("  python f1_predictor.py [OPTION]")
    print("\nOPTIONS:")
    print("  (no argument)   Run complete pipeline (data → features → simulation)")
    print("  --data          Download/update F1 data only")
    print("  --features      Engineer features from existing data only")
    print("  --simulation    Run simulation from existing features only")
    print("  --status        Show current data file status")
    print("  --help          Show this help message")
    print("\nEXAMPLES:")
    print("  python f1_predictor.py              # Full pipeline")
    print("  python f1_predictor.py --data       # Just download data")
    print("  python f1_predictor.py --features   # Just engineer features")
    print("  python f1_predictor.py --simulation # Just run simulation")
    print("\nWORKFLOW:")
    print("  1. First time: Run without arguments for complete pipeline")
    print("  2. Updates: Use --data to refresh, then --features and --simulation")
    print("  3. Testing: Run individual steps to debug or experiment")
    print("\nFILES:")
    print("  • data_ingestion.py      - Downloads F1 data using FastF1")
    print("  • feature_engineering.py - Creates ML features from data")
    print("  • simulation_engine.py   - Trains model & runs Monte Carlo predictions")
    print("\nREQUIRED PACKAGES:")
    print("  fastf1, pandas, numpy, scikit-learn")
    print("  Install with: pip install -r requirements.txt")

def main():
    parser = argparse.ArgumentParser(
        description="F1 Predictor - Predict Formula 1 race outcomes",
        add_help=False
    )
    parser.add_argument('--data', action='store_true', help='Run data ingestion only')
    parser.add_argument('--features', action='store_true', help='Run feature engineering only')
    parser.add_argument('--simulation', action='store_true', help='Run simulation only')
    parser.add_argument('--status', action='store_true', help='Show data status')
    parser.add_argument('--help', action='store_true', help='Show help message')

    args = parser.parse_args()

    if args.help:
        show_help()
        return 0

    if args.status:
        print_banner()
        print("F1 PREDICTOR DATA STATUS")
        show_data_status()
        return 0

    if args.data:
        return run_data_only()
    elif args.features:
        return run_features_only()
    elif args.simulation:
        return run_simulation_only()

    return run_full_pipeline()

if __name__ == "__main__":
    sys.exit(main())
