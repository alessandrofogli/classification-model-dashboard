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
                categorical_features = st.multiselect("Select categorical features (excluding target)", 
                                                      [col for col in df.columns if col != target_column])
                if st.button("Convert to Binary and Process"):
                    df = preprocess_data(df, target_column, option)
                    st.success(f"Converted {target_column} to binary with '{option}' as positive class (1).")
                    process_data(df, target_column, categorical_features)
            else:
                st.error("The selected target variable does not have enough unique values; please select a different column.")

def preprocess_data(df, target_column, positive_class):
    """Converts a selected category to 1 and all others to 0."""
    df[target_column] = df[target_column].apply(lambda x: 1 if x == positive_class else 0)
    return df

def process_data(df, target_column, categorical_features):
    # Define feature columns dynamically based on the target column
    feature_columns = [col for col in df.columns if col != target_column]
    num_cols = [col for col in feature_columns if col not in categorical_features]
    cat_cols = categorical_features

    # Display the selected numerical and categorical features
    st.write("Numerical features:", num_cols)
    st.write("Categorical features:", cat_cols)

    # Build and apply column transformer
    col_trans = build_column_transformer(num_cols, cat_cols, target_column)
    X = df[num_cols + cat_cols + [target_column]]
    y = df[target_column]
    transformed_data = col_trans.fit_transform(X, y)

    st.write("Data processing successful!")
    st.dataframe(transformed_data)  # Assuming transformed_data is a numpy array or DataFrame

if __name__ == "__main__":
    main()

