# model_utils.py
import pickle
import streamlit as st

def save_model(model, file_name):
    """
    Serialize the model to a pickle file and create a download link in Streamlit.

    Args:
    model (model object): The trained model to be saved.
    file_name (str): The name of the file to save the model to.

    Returns:
    None, but creates a button in Streamlit for downloading the model.
    """
    # Serialize the model using pickle
    with open(file_name, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate a download link
    with open(file_name, 'rb') as file:
        st.download_button(
            label="Download Trained Model",
            data=file,
            file_name=file_name,
            mime="application/octet-stream"
        )
