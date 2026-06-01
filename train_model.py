import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def prepare_features(df):
    """Prepare features for machine learning model."""
    # Create a copy to avoid modifying original data
    data = df.copy()

    # Handle missing values
    # For qualifying position, fill with grid position or a high value
    data['quali_position'] = data['quali_position'].fillna(data['grid_position'])
    data['quali_position'] = data['quali_position'].fillna(20)  # Default to back of grid

    # For Q1, Q2, Q3 times, we'll convert them to seconds or use qualifying position
    # For now, we'll focus on quali_position as our main qualifying feature

    # Encode categorical variables
    le_driver = LabelEncoder()
    le_constructor = LabelEncoder()
    le_circuit = LabelEncoder()

    data['driver_encoded'] = le_driver.fit_transform(data['driver_id'])
    data['constructor_encoded'] = le_constructor.fit_transform(data['constructor_id'])
    data['circuit_encoded'] = le_circuit.fit_transform(data['circuit_id'])

    # Feature engineering
    # Experience: we could use round number as a proxy for season progress
    # Team performance: constructor points so far (we'd need to calculate this)
    # For simplicity, we'll use what we have

    features = [
        'quali_position',
        'grid_position',
        'round',
        'driver_encoded',
        'constructor_encoded',
        'circuit_encoded'
    ]

    # Check if all features exist
    available_features = [f for f in features if f in data.columns]
    print(f"Using features: {available_features}")

    X = data[available_features]
    y = data['finish_position']  # What we want to predict

    return X, y, le_driver, le_constructor, le_circuit, available_features

def train_model(X, y):
    """Train a Random Forest model to predict finish position."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"Training MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)

    return model, X_test, y_test, y_pred_test

def save_model(model, label_encoders, feature_names, model_dir="/home/bobo/Projects/f1_predictor/models"):
    """Save the trained model and encoders."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "f1_race_position_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save encoders
    encoder_path = os.path.join(model_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoder_path)
    print(f"Label encoders saved to {encoder_path}")

    # Save feature names
    features_path = os.path.join(model_dir, "feature_names.pkl")
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to {features_path}")

def main():
    """Main function to load data, train model, and save it."""
    # Load the combined dataset
    data_path = "/home/bobo/Projects/f1_predictor/data/processed/combined_dataset.csv"
    print(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")

    # Prepare features
    X, y, le_driver, le_constructor, le_circuit, feature_names = prepare_features(df)

    # Train model
    model, X_test, y_test, y_pred_test = train_model(X, y)

    # Save model and encoders
    label_encoders = {
        'driver': le_driver,
        'constructor': le_constructor,
        'circuit': le_circuit
    }

    save_model(model, label_encoders, feature_names)

    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)

    # Show some actual vs predicted values
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred_test
    })
    comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']
    comparison_df = comparison_df.head(10)

    print("Sample predictions (first 10 test samples):")
    print(comparison_df.to_string(index=False))

    # Calculate accuracy within +/- 1 position
    accurate_predictions = np.abs(y_test - y_pred_test) <= 1
    accuracy = np.mean(accurate_predictions) * 100
    print(f"\nAccuracy (±1 position): {accuracy:.1f}%")

if __name__ == "__main__":
    main()