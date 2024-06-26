import streamlit as st
from sklearn.model_selection import train_test_split
import logging
import sys
import os

from data_processing.data_loader import load_data, preprocess_data
from data_processing.preprocessing import build_column_transformer_woe, build_column_transformer
from models.model_train import train_model
from models.model_evaluation import evaluate_model
from models.model_utils import save_model
from statistics_plots.data_visualization import plot_roc_curve

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def upload_data():
    st.title("Upload your CSV data")
    
    # Provide a button to load the sample data
    if st.button('Load Sample Data'):
        # Assuming the data folder is in the same directory as your app.py
        sample_data_path = os.path.join('data', 'german_credit_data.csv')
        df = load_data(sample_data_path)
        st.session_state['df'] = df
        st.write("Sample data loaded. First 5 rows of the sample data:")
        st.dataframe(df.head())
        reset_model_state()
    
    # Allow user to upload their own file
    uploaded_file = st.file_uploader("Or upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state['df'] = df
        st.write("Uploaded data. First 5 rows of your data:")
        st.dataframe(df.head())
        reset_model_state()

    # Ensure there's a way to proceed
    if 'df' in st.session_state and not st.session_state['df'].empty:
        if st.button("Next"):
            st.session_state['current_stage'] = "Feature Engineering"
            st.experimental_rerun()

def reset_model_state():
    # Resets the training related states
    if 'model_trained' in st.session_state:
        del st.session_state['model_trained']
    if 'model' in st.session_state:
        del st.session_state['model']

def feature_engineering():
    st.title("Feature Engineering")
    df = st.session_state.get('df', None)
    if df is not None:
        target_column = st.selectbox("Select the target variable (If you loaded the data sample use 'Risk')", df.columns)
        unique_values = df[target_column].dropna().unique()
        logging.info(f"Unique values in target column: {unique_values}")

        if len(unique_values) == 2:
            option = st.selectbox("Select the event class for conversion (will be converted to 1 --> If you loaded the data sample use 'Bad')", unique_values)
            categorical_features = st.multiselect("Select categorical features (If you loaded the data sample select:  Sex, Job, Housing, Saving accounts, Checking account, Purpose)",
                                                  [col for col in df.columns if col != target_column])
            
            # Allow user to select the preprocessing method
            preprocessing_method = st.selectbox("Select preprocessing method:",
                                                ["WoE", "StandardScaler + OneHotEncoding"])
            st.session_state['preprocessing_method'] = preprocessing_method

            if st.button("Convert to Binary and Process"):
                df = preprocess_data(df, target_column, option)
                st.session_state['df'] = df
                st.session_state['target_column'] = target_column
                st.session_state['categorical_features'] = categorical_features
                st.session_state['current_stage'] = "Model Training and Evaluation"
                st.rerun()
        else:
            st.error("Only binary classification is supported; please select a different column.")

def model_training():
    st.title("Model Training and Evaluation")
    df = st.session_state.get('df', None)
    target_column = st.session_state.get('target_column', None)
    categorical_features = st.session_state.get('categorical_features', [])

    # Only process and train if the model has not been trained yet or needs to be retrained
    if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
        if df is not None and target_column is not None:
            process_and_train(df, target_column, categorical_features)

    # Check if the model is successfully trained
    if 'model' in st.session_state and st.session_state['model'] is not None:
        st.success("Model training complete! Proceed to download your model.")
        if st.button("Next - Go to Model Download"):
            st.session_state['current_stage'] = "Model Download"
            st.session_state['model_trained'] = True  # Ensure training does not re-trigger
            st.rerun()


def model_download():
    st.title("Download Trained Model")
    model = st.session_state.get('model', None)
    if model is not None:
        save_model(model, 'trained_model.pkl')
    else:
        st.write("No model available to download. Please train a model first.")

def process_and_train(df, target_column, categorical_features=[]):
    
    logging.info("Starting process and train function.")
    feature_columns = [col for col in df.columns if col != target_column]
    num_cols = [col for col in feature_columns if col not in categorical_features]
    cat_cols = categorical_features

    # Select the column transformer based on user choice
    if st.session_state['preprocessing_method'] == "WoE":
        col_trans = build_column_transformer_woe(num_cols, cat_cols, target_column)
    else:
        col_trans = build_column_transformer(num_cols, cat_cols)
    
    # Split data
    X = df[num_cols + cat_cols + ([target_column] if st.session_state['preprocessing_method'] == "WoE" else [])]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("Data split into train and test sets.")

    with st.spinner('Training model... Please wait'):
        try:
            gs = train_model(X_train, y_train, col_trans)
            evaluation_results = evaluate_model(gs, X_test, y_test)
            st.session_state['model'] = gs.best_estimator_
            st.session_state['model_trained'] = True
            if evaluation_results:
                st.write("Evaluation Results:", evaluation_results)
                st.success('Model training complete!')
                logging.info("Model trained successfully.")
            else:
                st.error("Failed to compute evaluation metrics.")
            logging.info("Model trained successfully.")
        except Exception as e:
            st.error("An error occurred during model training: " + str(e))
            logging.error("Error during model training: ", exc_info=True)
            st.session_state['model_trained'] = False

    # Plot ROC curve if model training is successful
    y_scores = gs.predict_proba(X_test)[:, 1]
    fig = plot_roc_curve(y_test, y_scores)
    st.pyplot(fig)
    logging.info("ROC curve plotted successfully.")

def main():
    st.sidebar.title("Machine Learning Dashboard")
    # Define the stages
    app_modes = ["Upload Data", "Feature Engineering", "Model Training and Evaluation", "Model Download"]
    # Bind the sidebar's state to a session state variable
    if 'current_stage' not in st.session_state:
        st.session_state['current_stage'] = app_modes[0]  # Default to the first mode
    
    # Radio button that's bound to the session state variable
    st.session_state['current_stage'] = st.sidebar.radio("Choose the stage", app_modes, index=app_modes.index(st.session_state['current_stage']))

    # Navigation based on the current stage set in session state
    if st.session_state['current_stage'] == "Upload Data":
        upload_data()
    elif st.session_state['current_stage'] == "Feature Engineering":
        feature_engineering()
    elif st.session_state['current_stage'] == "Model Training and Evaluation":
        model_training()
    elif st.session_state['current_stage'] == "Model Download":
        model_download()


if __name__ == "__main__":
    main()
