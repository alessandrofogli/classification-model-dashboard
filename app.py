import streamlit as st
from sklearn.model_selection import train_test_split
import logging
import sys

from data_processing.data_loader import load_data, preprocess_data
from data_processing.preprocessing import build_column_transformer
from models.model_train import train_model
from models.model_evaluation import evaluate_model
from models.model_utils import save_model
from statistics_plots.data_visualization import plot_roc_curve

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    st.title("Machine Learning Classification Dashboard")

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        logging.info("File uploaded successfully.")
        df = load_data(uploaded_file)
        st.write("First 5 rows of your data:")
        st.dataframe(df.head())

        if not df.empty:
            target_column = st.selectbox("Select the target variable", df.columns)
            unique_values = df[target_column].dropna().unique()
            logging.info(f"Unique values in target column: {unique_values}")

            if set(unique_values) == {0, 1} or set(unique_values) == {1, 0}:
                process_and_train(df, target_column)
            elif len(unique_values) == 2:
                option = st.selectbox("Select the event class for conversion (will be converted to 1)", unique_values)
                categorical_features = st.multiselect("Select categorical features (excluding target)",
                                                      [col for col in df.columns if col != target_column])
                if st.button("Convert to Binary and Process"):
                    df = preprocess_data(df, target_column, option)
                    st.success(f"Converted {target_column} to binary with '{option}' as positive class (1).")
                    process_and_train(df, target_column, categorical_features)
            else:
                st.error("The selected target variable has too many unique values; only binary classification is supported; please select a different column.")
                logging.warning("Too many unique values in target column for binary classification.")


def process_and_train(df, target_column, categorical_features=[]):
    
    logging.info("Starting process and train function.")
    feature_columns = [col for col in df.columns if col != target_column]
    num_cols = [col for col in feature_columns if col not in categorical_features]
    cat_cols = categorical_features

    #st.write("Numerical features:", num_cols)
    #st.write("Categorical features:", cat_cols)

    col_trans = build_column_transformer(num_cols, cat_cols, target_column)
    X = df[num_cols + cat_cols + [target_column]]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Data split into train and test sets.")

    #transformed_X_train = col_trans.fit_transform(X_train, y_train)
    #transformed_X_test = col_trans.transform(X_test)

    #st.write("Data processing successful!")
    #st.dataframe(transformed_X_train)  # Assuming transformed_data is a numpy array or DataFrame
    with st.spinner('Training model... Please wait'):
        try:
            gs = train_model(X_train, y_train, col_trans)
            evaluation_results = evaluate_model(gs, X_test, y_test)
            st.write("Evaluation Results:", evaluation_results)
            st.success('Model training complete!')
            logging.info("Model trained successfully.")
        except Exception as e:
            st.error("An error occurred during model training.")
            logging.error("Error during model training: ", exc_info=True)


    y_scores = gs.predict_proba(X_test)[:,1]
    fig = plot_roc_curve(y_test, y_scores)
    st.pyplot(fig)
    logging.info("ROC curve plotted successfully.")
    save_model(gs.best_estimator_, 'trained_model.pkl')
    logging.info("Trained model saved successfully.")


if __name__ == "__main__":
    main()
