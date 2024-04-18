import streamlit as st
from data_processing.data_loader import load_data, preprocess_data
from data_processing.preprocessing import build_column_transformer

def main():
    st.title("Machine Learning Classification Dashboard")

    # File uploader allows user to add their own CSV
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])


    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.write("Columns in DataFrame after preprocessing:", df.columns.tolist())

            df_preprocessed = preprocess_data(df)

            
            # Log column names to check them
            st.write("Columns in the processed DataFrame:", df_preprocessed.columns.tolist())

            y = df_preprocessed['Risk']  # Ensure 'Risk' is the correct name
            X = df_preprocessed
            
            num_cols = ['Age', 'Credit amount', 'Duration']  # Example numeric columns
            cat_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']  # Example categorical columns
 
            # Check if the expected columns are in X
            expected_columns = set(num_cols + cat_cols)
            if not expected_columns.issubset(X.columns):
                missing_cols = expected_columns - set(X.columns)
                st.error(f"Missing columns: {missing_cols}")
                return  # Stop further execution

            col_trans = build_column_transformer(num_cols, cat_cols)
            transformed_data = col_trans.fit_transform(X, y)

            st.write("Data processing successful!")
            st.dataframe(transformed_data)

        except Exception as e:
            st.error(f"Error processing the data: {e}")


if __name__ == "__main__":
    main()
