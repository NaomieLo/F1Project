# F1 Race Predictor Bot

An AI-powered tool that predicts Formula 1 race results by analyzing historical correlations between Qualifying performance and Race Day outcomes.

# About The Project

This project uses machine learning to forecast the finishing order of drivers before the race begins. It operates in three stages:

## Data Harvesting

First, the bot connects to official F1 timing servers via the FastF1 API. It pulls historical data—specifically looking at Qualifying lap times, starting grid positions, and weather conditions (like rain or track temperature).

## Pattern Recognition

A machine learning model (Random Forest Regressor) is trained on this historical data. Think of it like a "super-fan" that has memorized every race from the last few years. It learns patterns—for example, 'How often does the driver on Pole Position actually win?' or 'How much does rain affect the result?'

## Prediction

Before a race starts, the bot takes the current weekend's fresh qualifying data and runs it through its 'memory' of past races to generate a predicted finishing order for every driver.

# Prerequisites

Due to compatibility issues with scientific libraries (NumPy/Pandas) on newer Python versions, this project requires Python 3.11 or 3.12.

Python 3.11 (Recommended)

MacOS/Linux (Commands below are optimized for these systems)

# Installation & Setup

1. Create a virtual environment (Python 3.11)

`
python3.11 -m venv venv 2. Activate the environment

source venv/bin/activate 3. Confirm Python Version Ensure you are running a stable version (Output should be Python 3.11.x).

python --version 4. Install Dependencies This installs fastf1, pandas, scikit-learn, and other required tools.

pip install --upgrade pip
pip install fastf1 pandas numpy scikit-learn joblib
`

# Usage

1. Train the Model
   You must train the model first. This script fetches historical data, trains the Random Forest, and saves the model as a .pkl file.

`python f1_train.py`

Output: This will generate f1_model_post_qualifying.pkl.

2. Make a Prediction
   Once the model is trained, you can generate predictions for an upcoming race.

Important: This script requires Qualifying data. You should run this after Qualifying sessions have concluded on Saturday.

Open f1_predict.py and ensure the predict_gp_name matches the current race weekend (e.g., "Abu Dhabi Grand Prix").

Run the predictor:
`python f1_predict.py`

# Project Structure

f1_data.py: Handles caching and API connections to FastF1.

f1_features.py: Engineering logic to convert raw lap times into usable ML features.

f1_train.py: Main training pipeline. Fetches history, trains model, saves .pkl.

f1_predict.py: Loads the saved model and predicts the outcome of the specific target race.
