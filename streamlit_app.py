# streamlit_app.py
import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Import the recommendation system directly
from recommendation_system import SHLRecommendationSystem

# Load environment variables
load_dotenv()

# Configure Google API
google_api_key = os.environ.get("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
# In streamlit_app.py, replace the start_api_server function
def start_api_server():
    try:
        # On Streamlit Cloud, don't try to start the API server
        if os.environ.get("IS_STREAMLIT_CLOUD") == "1":
            return None
            
        # For local development only:
        # Determine the correct path to api_server.py
        api_server_path = os.path.join(os.path.dirname(__file__), 'api_server.py')
        
        # Find an available port
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
        s.close()
        
        # Set environment variable for the API to use
        os.environ["PORT"] = str(port)
        
        # Start the API server as a subprocess with the dynamic port
        api_process = subprocess.Popen([sys.executable, api_server_path])
        
        # Wait a moment to ensure the server starts
        time.sleep(5)
        
        # Set the API URL to use the dynamic port
        global API_URL
        API_URL = f"http://localhost:{port}"
        
        return api_process
    except Exception as e:
        st.error(f"Failed to start API server: {e}")
        return None

# Initialize recommendation system
@st.cache_resource
def get_recommendation_system():
    try:
        processed_data_path = "FinalDataSource/processed_assessments.json"
        db_directory = "chroma_db"
        recommendation_system = SHLRecommendationSystem(
            processed_data_path=processed_data_path,
            db_directory=db_directory
        )
        return recommendation_system
    except Exception as e:
        st.error(f"Failed to initialize recommendation system: {e}")
        return None

# Configure Streamlit page
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("SHL Assessment Recommendation System")
st.markdown("""
This application helps hiring managers find the right assessments for their roles.
Simply enter a job description or query, and get tailored SHL assessment recommendations.
""")

# Example queries
example_queries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
    "Need technical assessment for senior software developers with focus on system design"
]

with st.sidebar:
    st.header("About")
    st.info("This tool uses natural language processing to recommend relevant SHL assessments based on your requirements.")
    
    st.header("Example Queries")
    selected_example = st.selectbox("Try an example query:", [""] + example_queries)
    
    st.header("Settings")
    max_results = st.slider("Maximum number of recommendations", min_value=1, max_value=10, value=5)

# Query input
if selected_example:
    query = st.text_area("Enter your job description or query:", value=selected_example, height=150)
else:
    query = st.text_area("Enter your job description or query:", 
                         placeholder="Example: Looking for an assessment to evaluate Python programming skills for entry-level developers",
                         height=150)

# Submit button
col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Get Recommendations", type="primary")

# Display recommendations if query submitted
if submit_button and query.strip():
    # Get recommendation system
    recommendation_system = get_recommendation_system()
    
    if recommendation_system:
        with st.spinner("Analyzing your requirements..."):
            try:
                # Get recommendations directly
                df = recommendation_system.get_recommendations(query, max_results=max_results)
                
                if df.empty or df.iloc[0]["Assessment Name"] == "No matching assessments found":
                    st.warning("No matching assessments found for your query.")
                else:
                    st.success(f"Found {len(df)} relevant assessments:")
                    
                    # Format URLs as clickable links
                    df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if x else '')
                    
                    # Display results in an interactive table
                    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Download as CSV option
                    csv_df = df.copy()
                    # Remove HTML tags for CSV download
                    csv_df['URL'] = csv_df['URL'].str.replace(r'<.*?>', '', regex=True)
                    csv = csv_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "shl_recommendations.csv",
                        "text/csv",
                        key='download-csv'
                    )
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
                import traceback
                st.error(traceback.format_exc())
    else:
        st.error("Recommendation system could not be initialized")
else:
    if submit_button:
        st.warning("Please enter a query to get recommendations.")
    else:
        st.info("Enter your requirements above and click 'Get Recommendations'.")

# Footer
st.markdown("---")
st.markdown("Powered by AI and natural language processing")
