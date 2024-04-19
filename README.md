 # Streamlit Machine Learning Classification Dashboard

This Streamlit application provides an intuitive dashboard for users to upload a dataset, select a target feature, train a machine learning model, evaluate its performance, and download the trained model for further use. It's designed to facilitate quick and efficient data-driven decision-making without requiring deep programming expertise.

## Features

### 1. Data Upload
- **Description**: Users can upload their dataset in a DataFrame format.
- **Supported Formats**: Primarily supports `.csv` files. More formats can be added as needed.

### 2. Feature Selection
- **Description**: After uploading the dataset, the application displays the names of all features (columns) in the dataset.
- **User Interaction**: Users can select the target feature for model training using a dropdown menu.

### 3. Statistics Display
- **Description**: Basic statistics about the dataset are displayed to help users understand the data distribution and quality.
- **Statistics Include**: Count, mean, standard deviation, min/max values, and other relevant descriptive statistics.

### 4. Model Training
- **Description**: Users can train a machine learning model with a simple click of a button.
- **Features Transformation**: Binning and WoE (Weight-of-Evidence) is applied. More preprocessing methos can be added later.
- **Models Supported**: Initially, XGBoost. More models can be added later.

### 5. Model Evaluation
- **Description**: After training, the model's performance is evaluated and various metrics are plotted.
- **Metrics Displayed**: Accuracy, Precision, Recall, ROC Curve, etc.

### 6. Model Download
- **Description**: Users can download the trained model in `.pkl` format.
- **Usage**: The downloaded model can be integrated into other applications or used for further analysis.

## Future Enhancements
- **Feature Importance**: Display feature importance scores for the trained models.
- **Advanced Model Training**: Options to customize hyperparameters and training algorithms.

## Setup and Installation
- **Requirements**: Python 3.8+, Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Installation**:
  ```bash
  pip install -r requirements.txt

```
classification-model-dashboard/
│
├── app.py                  # Main application file where Streamlit UI components are defined.
│
├── data_processing/
│   ├── __init__.py         # Makes data_processing a Python module.
│   └── data_loader.py      # Contains functions to load and preprocess the target variable.
│   └── preprocessing.py    # Contains classes and functions build column transformer for the features.
|
│
├── features/
│   ├── __init__.py         # Makes features a Python module.
│   └── feature_selector.py # Functions for selecting features and target from the dataset.
│
├── models/
│   ├── __init__.py         # Makes models a Python module.
│   ├── model_train.py      # Functions to train different machine learning models.
│   └── model_evaluation.py # Functions to evaluate models and compute metrics.
│
├── statistics/
│   ├── __init__.py         # Makes statistics a Python module.
│   └── descriptive_stats.py# Functions to compute and display descriptive statistics.
│
├── utils/
│   ├── __init__.py         # Makes utils a Python module.
│   └── file_manager.py     # Utility functions for file management like downloading models.
│
└── requirements.txt        # File containing all necessary Python packages.
```
