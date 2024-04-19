import streamlit as st
from data_processing.data_loader import load_data, preprocess_data
from data_processing.preprocessing import build_column_transformer

def main():
    st.title("Machine Learning Classification Dashboard")

    # File uploader allows user to add their own CSV
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Display the first 5 rows of the DataFrame
        st.write("First 5 rows of your data:")
        st.dataframe(df.head())

        # Dropdown for selecting the target variable
        if not df.empty:
            target_column = st.selectbox("Select the target variable", df.columns)
            unique_values = df[target_column].dropna().unique()

            # Check if the target is binary (0, 1 only)
            if set(unique_values) == {0, 1} or set(unique_values) == {1, 0}:
                if st.button("Process Data with Binary Target"):
                    process_data(df, target_column)
            elif len(unique_values) == 2:
                option = st.selectbox("Select the positive class for conversion", unique_values)
                if st.button("Convert to Binary and Process"):
                    df = preprocess_data(df, target_column, option)
                    st.success(f"Converted {target_column} to binary with '{option}' as positive class (1).")
                    process_data(df, target_column)
            else:
                st.error("The selected target variable does not have enough unique values; please select a different column.")

def process_data(df, target_column):
    # Define feature columns dynamically based on the target column
    feature_columns = [col for col in df.columns if col != target_column]

    # Build and apply column transformer
    col_trans = build_column_transformer(feature_columns, target_column)
    X = df[feature_columns]
    y = df[target_column]
    transformed_data = col_trans.fit_transform(X, y)

    st.write("Data processing successful!")
    st.dataframe(transformed_data)  # Assuming transformed_data is a numpy array or DataFrame

if __name__ == "__main__":
    main()
