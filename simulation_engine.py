import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = './data'
FEATURE_FILE = os.path.join(DATA_DIR, 'driver_features.csv')

def train_local_ml_model():
    """Trains a local Random Forest model to predict base race performance."""
    if not os.path.exists(FEATURE_FILE):
        print("Error: driver_features.csv not found. Run feature_engineering.py first.")
        return None
    
    df = pd.read_csv(FEATURE_FILE)
    
    # Define our ML features (X) and what we want to predict (y)
    feature_cols = ['GridPosition', 'Recent_Form', 'Track_History_Avg', 'Driver_Incident_Index']
    
    # Drop rows that have any missing values in our features
    df_clean = df.dropna(subset=feature_cols + ['FinalPosition'])
    
    X = df_clean[feature_cols]
    y = df_clean['FinalPosition']
    
    print(f"Training local Machine Learning model on {len(df_clean)} historical rows...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model training complete!")
    return model

def run_monte_carlo_simulation(model, race_grid, simulations=10000):
    """
    Runs a 10,000-race Monte Carlo simulation injecting statistical chaos
    based on each driver's custom Incident Index.
    """
    print(f"\nLaunching Monte Carlo engine ({simulations} simulated races)...")
    
    # Dictionary to keep track of how many times each driver finishes in what position
    driver_results = {driver: [] for driver in race_grid.keys()}
    
    # Loop 10,000 times
    for sim in range(simulations):
        simulated_positions = {}
        
        for driver, profiles in race_grid.items():
            # 1. Ask our local ML model to predict the "perfect world" base finish position
            input_data = pd.DataFrame([profiles['features']])
            base_prediction = model.predict(input_data)[0]
            
            # 2. Inject Chaos (The Monte Carlo element)
            # If a random number is lower than their incident index, they suffer a 'crash/issue' delay
            chaos_factor = 0
            if np.random.rand() < profiles['features']['Driver_Incident_Index']:
                # Driver gets hit with a heavy delay penalty (simulating a crash, spin, or DNF)
                chaos_factor = np.random.uniform(5, 15) 
            
            # Final simulated score for this specific race run
            simulated_positions[driver] = base_prediction + chaos_factor + np.random.normal(0, 1)

        # Sort the drivers for this specific simulated run based on who got the lowest score (best finish)
        sorted_race = sorted(simulated_positions.items(), key=lambda x: x[1])
        
        for rank, (driver, _) in enumerate(sorted_race):
            driver_results[driver].append(rank + 1)
            
    return driver_results

if __name__ == "__main__":
    # 1. Train the ML brain
    ml_model = train_local_ml_model()
    
    if ml_model is not None:
        # 2. Setup a hypothetical grid for testing (e.g., matching the sample data structures)
        # Features: ['GridPosition', 'Recent_Form', 'Track_History_Avg', 'Driver_Incident_Index']
        mock_2026_grid = {
            'RIC': {'features': {'GridPosition': 1, 'Recent_Form': 5.0, 'Track_History_Avg': 6.0, 'Driver_Incident_Index': 0.02}},
            'ALB': {'features': {'GridPosition': 5, 'Recent_Form': 14.0, 'Track_History_Avg': 18.0, 'Driver_Incident_Index': 0.17}},
            'SAI': {'features': {'GridPosition': 2, 'Recent_Form': 6.5, 'Track_History_Avg': 8.0, 'Driver_Incident_Index': 0.06}},
            'BOT': {'features': {'GridPosition': 10, 'Recent_Form': 15.0, 'Track_History_Avg': 12.0, 'Driver_Incident_Index': 0.09}}
        }
        
        # 3. Simulate!
        raw_sim_data = run_monte_carlo_simulation(ml_model, mock_2026_grid)
        
        # 4. Calculate Probabilities for presentation
        print("\n=== MONTE CARLO PROBABILITY RESULTS ===")
        for driver, finishes in raw_sim_data.items():
            finishes_arr = np.array(finishes)
            win_pct = (np.sum(finishes_arr == 1) / len(finishes_arr)) * 100
            podium_pct = (np.sum(finishes_arr <= 3) / len(finishes_arr)) * 100
            avg_finish = np.mean(finishes_arr)
            
            print(f"Driver {driver} -> Win Probability: {win_pct:.2f}% | Podium Probability: {podium_pct:.2f}% | Avg Finish: {avg_finish:.1f}")