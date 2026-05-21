import os
import pandas as pd
import numpy as np

DATA_DIR = './data'
INPUT_FILE = os.path.join(DATA_DIR, 'historical_results_2021_2025.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'driver_features.csv')

def load_and_clean_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}. Please run data_ingestion.py first.")
        return None
    
    df = pd.read_csv(INPUT_FILE)
    
    # Clean positions: If a driver didn't finish, FastF1 might leave Position blank or as a string.
    # We force it to be numeric, and fill missing positions with 20 (assuming last place for DNFs)
    df['FinalPosition'] = pd.to_numeric(df['FinalPosition'], errors='coerce')
    df['FinalPosition'] = df['FinalPosition'].fillna(20).astype(int)
    
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['GridPosition'] = df['GridPosition'].fillna(20).astype(int)
    
    return df

def calculate_metrics(df):
    print("Calculating local mathematical features...")
    
    # Sort data chronologically so our rolling metrics make statistical sense
    df = df.sort_values(by=['Year', 'RoundNumber']).reset_index(drop=True)
    
    # 1. INCIDENT INDEX
    # We look at the 'Status' column. If it says 'Collision', 'Accident', or 'Spun off', we flag it.
    crash_keywords = ['collision', 'accident', 'spun', 'damage', 'crash']
    df['IsIncident'] = df['Status'].astype(str).str.lower().apply(
        lambda x: 1 if any(kw in x for kw in crash_keywords) else 0
    )
    
    # Calculate a running historical crash rate for each driver
    # (Total crashes divided by total races entered up to that point)
    df['Driver_Incident_Index'] = df.groupby('DriverAbbreviation')['IsIncident'].transform(
        lambda x: x.expanding().mean()
    )

    # 2. RECENT FORM (Rolling average of the last 5 races)
    df['Recent_Form'] = df.groupby('DriverAbbreviation')['FinalPosition'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # 3. TRACK TRACK-SPECIFIC HISTORY
    # Average finishing position of a driver at a specific track
    track_history = df.groupby(['DriverAbbreviation', 'Track'])['FinalPosition'].mean().reset_index()
    track_history.rename(columns={'FinalPosition': 'Track_History_Avg'}, inplace=True)
    
    # Merge track history back into the main dataframe
    df = pd.merge(df, track_history, on=['DriverAbbreviation', 'Track'], how='left')
    
    return df

if __name__ == "__main__":
    # Load raw data
    raw_df = load_and_clean_data()
    
    if raw_df is not None:
        # Transform data into ML features
        featured_df = calculate_metrics(raw_df)
        
        # Save the structured feature dataset
        featured_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Featured dataset saved to {OUTPUT_FILE}")
        
        # Print a quick preview so you can check the math
        print("\n--- Feature Preview (Sample Data) ---")
        print(featured_df[['Year', 'Track', 'DriverAbbreviation', 'Recent_Form', 'Driver_Incident_Index', 'Track_History_Avg']].tail(10))