# Remote Health Agent

The Remote Health Agent is an AI-based system designed to collect patient symptom data from remote locations, analyze the symptoms, and detect potential diseases. It utilizes machine learning algorithms and historical data to enhance its disease detection capabilities.

## Features

- Collects patient symptom data from a CSV file.
- Analyzes symptoms for disease detection using machine learning.
- Builds a graph representing relationships between symptoms and diseases.
- Integrates with healthcare systems for further diagnosis and treatment recommendations.

## Usage

1. Install required dependencies:
pip install -r requirements.txt


3. Input a patient ID when prompted to analyze symptoms for the provided patient.

## Files

- `remote_health_agent.py`: Main Python script containing the RemoteHealthAgent class implementation.
- `generator_csv.py`: Python script to generate sample CSV files containing patient symptom data.
- `patient_data.csv`: Sample CSV file containing patient symptom data.

## Requirements

- Python 3.x
- scikit-learn
- numpy
