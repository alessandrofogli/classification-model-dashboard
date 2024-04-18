import streamlit as st
from data_processing.data_loader import load_data, preprocess_data
from data_processing.preprocessing import build_column_transformer

def main():
    st.title("Machine Learning Classification Dashboard")

    # File uploader allows user to add their own CSV
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load and preprocess the data
        df = load_data(uploaded_file)
        df_preprocessed = preprocess_data(df)
        
        # Display the first 5 rows of the DataFrame
        st.write("First 5 rows of your data:")
        st.dataframe(df_preprocessed.head())

        # Dropdown for selecting the target variable
        if not df_preprocessed.empty:
            target_column = st.selectbox("Select the target variable", df_preprocessed.columns)

            if st.button("Process Data"):
                try:
                    st.write("You selected:", target_column)

                    y = df_preprocessed[target_column]
                    X = df_preprocessed

                    # Assuming 'num_cols' and 'cat_cols' are defined; ensure 'target_column' is not in 'num_cols' or 'cat_cols'
                    num_cols = [col for col in ['Age', 'Credit amount', 'Duration'] if col != target_column]  # Example numeric columns
                    cat_cols = [col for col in ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'] if col != target_column]  # Example categorical columns

                    # Building and applying column transformer
                    col_trans = build_column_transformer(num_cols, cat_cols, target_col=target_column)
                    transformed_data = col_trans.fit_transform(X, y)  # Make sure to pass y here

                    st.write("Data processing successful!")
                    st.dataframe(transformed_data)  # Assuming transformed_data is a numpy array or DataFrame

                except Exception as e:
                    st.error(f"Error processing the data: {e}")

if __name__ == "__main__":
    main()
