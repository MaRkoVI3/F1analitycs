                                                                             
                            F1 PREDICTOR - ANALYTICS CORE

PROJECT OVERVIEW
================

The F1 Predictor Analytics Core is a sophisticated machine learning and simulation system 
designed to predict Formula 1 race outcomes. It combines historical data analysis, 
feature engineering, and Monte Carlo simulation to generate accurate race predictions.

KEY FEATURES
============

• Historical Data Processing: Downloads and processes F1 data from 2021-2025 seasons
• Advanced Feature Engineering: Creates proprietary features including:
    - Driver Incident Index (measurement of driver volatility)
    - Recent Form (performance trends over last 3-5 races)
    - Track History Averages (driver/circuit specific performance)
    - Team Performance Trends
    - Car Development Rate (season-long performance evolution)
• Machine Learning Prediction: Uses Random Forest Regressor trained on historical data
• Monte Carlo Simulation: Runs 10,000 simulated races incorporating statistical chaos
• Probability Output: Generates win probabilities, podium chances, and expected finishes
• Modular Architecture: Separate components for data, features, and simulation

COMPONENTS
==========

1. data_ingestion.py
   - Downloads historical F1 data from FastF1 API
   - Collects race results, qualifying data, and telemetry
   - Stores data in structured CSV format for processing

2. feature_engineering.py  
   - Processes raw data into ML-ready features
   - Implements proprietary algorithms for Incident Index calculation
   - Creates track-specific and driver-specific performance metrics
   - Handles missing data and categorical variables

3. simulation_engine.py
   - Trains Machine Learning model on engineered features
   - Runs Monte Carlo simulations with injected statistical variance
   - Calculates win/podium/points probabilities for each driver
   - Outputs prediction confidence intervals

USAGE
=====

From the F1analitycs directory:
    python f1_predictor.py          # Run complete pipeline
    python f1_predictor.py --data   # Download data only
    python f1_predictor.py --features # Feature engineering only
    python f1_predictor.py --simulation # Run simulation only
    python f1_predictor.py --status   # Check data files

OUTPUT
======

Prediction results include:
- Win Probability (%)
- Podium Probability (%) 
- Average Expected Finish Position
- Confidence Intervals for each metric
- Driver-specific risk factors (Incident Index)

TECHNICAL SPECS
===============

• Language: Python 3.10+
• Libraries: pandas, numpy, scikit-learn, fastf1, joblib
• Data Sources: FastF1 (official F1 data provider)
• Model Type: Random Forest Regressor
• Simulation: 10,000 Monte Carlo iterations per race
• Features: 15+ engineered features including temporal and categorical variables

DATA FLOW
=========

Raw Data (JSON/CSV) 
        → Feature Engineering 
        → ML Features (CSV)
        → Model Training 
        → Trained Model (PKL)
        → Monte Carlo Simulation
        → Prediction Probabilities

NOTES
=====

• Requires internet connection for initial data download
• First run may take 10-15 minutes for data acquisition
• Subsequent runs are faster (2-5 minutes) using cached data
• Predictions are probabilistic - treat as guidance, not guarantees
• Model accuracy improves with more historical data

                                                                             
                            F1 PREDICTOR - ANALYTICS CORE
                                                                             
