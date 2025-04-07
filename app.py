# app.py
import streamlit.web.bootstrap
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Run the Streamlit app
if __name__ == "__main__":
    streamlit.web.bootstrap.run("streamlit_app", "", [], [])
