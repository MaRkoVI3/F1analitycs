#!/usr/bin/env python3
"""
F1 Predictor Menu System
Provides a unified interface to run the F1 prediction pipeline.
"""

import subprocess
import sys
import os

def print_menu():
    print("\n" + "="*50)
    print("F1 PREDICTOR MENU")
    print("="*50)
    print("1. Fetch latest F1 data from API (2021-2025)")
    print("2. Process raw data into features")
    print("3. Train prediction model")
    print("4. Predict current season races")
    print("5. View prediction results")
    print("6. Run full pipeline (1→2→3→4)")
    print("0. Exit")
    print("="*50)

def run_script(script_name, description):
    """Run a Python script and show output."""
    print(f"\n{description}...")
    print("-" * 50)
    try:
        # Run the script and capture output
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        result = subprocess.run([sys.executable, script_path],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"\n{description} completed.")
        if result.returncode != 0:
            print(f"Warning: Script exited with code {result.returncode}")
    except Exception as e:
        print(f"Error running {script_name}: {e}")

def main():
    while True:
        print_menu()
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
        except EOFError:
            # Handle non-interactive environment: run prediction by default and exit
            print("\nNo input available. Running prediction (option 4) by default.")
            run_script('predict_season.py', 'Generating predictions')
            break

        if choice == '0':
            print("\nExiting F1 Predictor. Goodbye! 🏎️")
            break
        elif choice == '1':
            run_script('fetch_data.py', 'Fetching F1 data')
        elif choice == '2':
            run_script('process_data.py', 'Processing data')
        elif choice == '3':
            run_script('train_model.py', 'Training model')
        elif choice == '4':
            run_script('predict_season.py', 'Generating predictions')
        elif choice == '5':
            view_results()
        elif choice == '6':
            run_full_pipeline()
        else:
            print("\nInvalid choice. Please enter a number between 0 and 6.")

def view_results():
    """View the latest prediction results."""
    print("\n" + "="*50)
    print("VIEWING PREDICTION RESULTS")
    print("="*50)
    predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'predictions')
    if not os.path.exists(predictions_dir):
        print("No predictions directory found. Run predictions first.")
        return

    # List available prediction files
    import glob
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
            # Show first 20 lines
            with open(selected_file, 'r') as f:
                lines = f.readlines()
                for line in lines[:20]:
                    print(line.rstrip())
                if len(lines) > 20:
                    print(f"... and {len(lines) - 20} more lines")
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
        ('fetch_data.py', 'Fetching F1 data'),
        ('process_data.py', 'Processing data'),
        ('train_model.py', 'Training model'),
        ('predict_season.py', 'Generating predictions')
    ]

    for script, description in steps:
        print(f"\nStep: {description}")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
        result = subprocess.run([sys.executable, script_path],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"Pipeline failed at step: {description}")
            print(f"Exit code: {result.returncode}")
            return

    print("\n" + "="*50)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY! 🏁")
    print("="*50)

if __name__ == "__main__":
    main()