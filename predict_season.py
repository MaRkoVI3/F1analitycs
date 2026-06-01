import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def load_model_and_encoders(model_dir="/home/bobo/Projects/f1_predictor/models"):
    """Load the trained model and encoders."""
    model_path = os.path.join(model_dir, "f1_race_position_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoders.pkl")
    features_path = os.path.join(model_dir, "feature_names.pkl")

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)

    return model, encoders, feature_names

def predict_race_outcomes_for_season(model, encoders, feature_names, season_data, season_year):
    """
    Predict race outcomes for a given season using qualifying data.
    Returns a DataFrame with predicted results for each race.
    """
    # Filter data for the given season
    season_data = season_data[season_data['season'] == season_year].copy()

    if len(season_data) == 0:
        print(f"No data found for season {season_year}")
        return None

    # We'll store predictions for each race
    predictions = []

    # Get unique races in the season
    races = season_data[['season', 'round', 'race_name', 'circuit_id', 'circuit_name', 'country', 'locality']].drop_duplicates()

    for _, race in races.iterrows():
        race_id = (race['season'], race['round'])
        # Get all drivers for this race
        race_drivers = season_data[
            (season_data['season'] == race['season']) &
            (season_data['round'] == race['round'])
        ].copy()

        if len(race_drivers) == 0:
            continue

        # Prepare features for prediction
        # We'll use qualifying position as grid position (since grid is set by qualifying)
        # For missing quali position, we'll use a high value (back of grid)
        race_drivers['quali_position'] = race_drivers['quali_position'].fillna(20)
        race_drivers['grid_position'] = race_drivers['quali_position']  # Assume grid equals quali

        # Encode categorical variables
        try:
            race_drivers['driver_encoded'] = encoders['driver'].transform(race_drivers['driver_id'])
        except ValueError as e:
            # Handle unseen drivers by assigning a default value (e.g., -1 or new class)
            # For simplicity, we'll assign the most frequent class or a default
            # In a real scenario, we might need to handle new drivers differently
            print(f"Warning: Unseen driver in {race['race_name']}. Using default encoding.")
            # Assign a default value (we'll use 0 for now, but this is not ideal)
            race_drivers['driver_encoded'] = 0

        try:
            race_drivers['constructor_encoded'] = encoders['constructor'].transform(race_drivers['constructor_id'])
        except ValueError as e:
            print(f"Warning: Unseen constructor in {race['race_name']}. Using default encoding.")
            race_drivers['constructor_encoded'] = 0

        try:
            race_drivers['circuit_encoded'] = encoders['circuit'].transform(race_drivers['circuit_id'])
        except ValueError as e:
            print(f"Warning: Unseen circuit in {race['race_name']}. Using default encoding.")
            race_drivers['circuit_encoded'] = 0

        # Select features for prediction
        X_race = race_drivers[feature_names]

        # Predict finish position
        predicted_positions = model.predict(X_race)
        race_drivers['predicted_position'] = predicted_positions

        # Sort by predicted position to get predicted order
        race_drivers_sorted = race_drivers.sort_values('predicted_position')

        # Add race info to each driver prediction
        for _, driver in race_drivers_sorted.iterrows():
            pred_record = {
                'season': driver['season'],
                'round': driver['round'],
                'race_name': driver['race_name'],
                'driver_id': driver['driver_id'],
                'driver_name': driver['driver_name'],
                'constructor_name': driver['constructor_name'],
                'quali_position': driver['quali_position'],
                'grid_position': driver['grid_position'],
                'predicted_position': driver['predicted_position'],
                'actual_position': driver['finish_position'],
                'points': 0  # We'll calculate points based on predicted position
            }
            predictions.append(pred_record)

    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Calculate points based on predicted position (F1 points system)
    points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # Points for positions 1-10
    def calculate_points(position):
        pos_int = int(round(position))  # Round to nearest integer
        if 1 <= pos_int <= 10:
            return points_system[pos_int - 1]
        else:
            return 0

    predictions_df['predicted_points'] = predictions_df['predicted_position'].apply(calculate_points)

    return predictions_df

def calculate_championship_standings(predictions_df):
    """Calculate championship standings from race predictions."""
    # Group by driver and sum points
    championship = predictions_df.groupby(['driver_id', 'driver_name', 'constructor_name'])['predicted_points'].sum().reset_index()
    championship = championship.rename(columns={'predicted_points': 'total_points'})
    championship = championship.sort_values('total_points', ascending=False)
    championship['position'] = range(1, len(championship) + 1)
    return championship

def main():
    """Main function to load model, make predictions for a season, and show results."""
    print("Loading model and encoders...")
    model, encoders, feature_names = load_model_and_encoders()
    print(f"Model loaded. Features: {feature_names}")

    # Load the combined dataset
    data_path = "/home/bobo/Projects/f1_predictor/data/processed/combined_dataset.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")

    # Predict for the most recent season we have (2025)
    season_to_predict = 2025
    print(f"\nPredicting season {season_to_predict}...")

    predictions_df = predict_race_outcomes_for_season(model, encoders, feature_names, df, season_to_predict)

    if predictions_df is None:
        print("Failed to generate predictions.")
        return

    print(f"Generated predictions for {predictions_df['race_name'].nunique()} races in {season_to_predict}")

    # Show some sample predictions
    print("\nSample predictions (first 5 races):")
    sample_races = predictions_df['race_name'].unique()[:5]
    for race in sample_races:
        race_preds = predictions_df[predictions_df['race_name'] == race].head(3)
        print(f"\n{race}:")
        for _, driver in race_preds.iterrows():
            print(f"  P{int(round(driver['predicted_position']))}: {driver['driver_name']} ({driver['constructor_name']}) - "
                  f"Quali: {driver['quali_position']}, Predicted: {driver['predicted_position']:.1f}, "
                  f"Actual: {driver['actual_position']}")

    # Calculate championship standings
    print("\n" + "="*60)
    print(f"PREDICTED CHAMPIONSHIP STANDINGS FOR {season_to_predict}")
    print("="*60)
    championship = calculate_championship_standings(predictions_df)
    print(championship[['position', 'driver_name', 'constructor_name', 'total_points']].to_string(index=False))

    # Calculate actual championship standings for comparison
    print("\n" + "="*60)
    print(f"ACTUAL CHAMPIONSHIP STANDINGS FOR {season_to_predict}")
    print("="*60)
    actual_championship = df[df['season'] == season_to_predict].groupby(
        ['driver_id', 'driver_name', 'constructor_name']
    )['points'].sum().reset_index()
    actual_championship = actual_championship.rename(columns={'points': 'total_points'})
    actual_championship = actual_championship.sort_values('total_points', ascending=False)
    actual_championship['position'] = range(1, len(actual_championship) + 1)
    print(actual_championship[['position', 'driver_name', 'constructor_name', 'total_points']].to_string(index=False))

    # Calculate accuracy of championship prediction
    # Compare top 3 drivers
    predicted_top3 = set(championship.head(3)['driver_name'])
    actual_top3 = set(actual_championship.head(3)['driver_name'])
    top3_accuracy = len(predicted_top3 & actual_top3) / 3 * 100
    print(f"\nTop 3 prediction accuracy: {top3_accuracy:.1f}%")

    # Overall championship position correlation (spearman rank correlation)
    from scipy.stats import spearmanr
    # Merge predicted and actual points for drivers that appear in both
    merged = pd.merge(
        championship[['driver_name', 'total_points']],
        actual_championship[['driver_name', 'total_points']],
        on='driver_name',
        suffixes=('_predicted', '_actual')
    )
    if len(merged) > 1:
        corr, _ = spearmanr(merged['total_points_predicted'], merged['total_points_actual'])
        print(f"Spearman correlation (championship points): {corr:.3f}")

    # Save predictions
    output_dir = "/home/bobo/Projects/f1_predictor/predictions"
    os.makedirs(output_dir, exist_ok=True)

    predictions_path = os.path.join(output_dir, f"{season_to_predict}_race_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nDetailed race predictions saved to {predictions_path}")

    championship_path = os.path.join(output_dir, f"{season_to_predict}_championship_prediction.csv")
    championship.to_csv(championship_path, index=False)
    print(f"Championship prediction saved to {championship_path}")

if __name__ == "__main__":
    main()