import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API URL from environment or use default
API_URL = os.getenv("API_URL", "http://localhost:8000")

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
    with st.spinner("Analyzing your requirements..."):
        try:
            # Call the API
            response = requests.get(
                f"{API_URL}/recommend",
                params={"query": query, "max_results": max_results},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])
                
                if not recommendations:
                    st.warning("No matching assessments found for your query.")
                else:
                    st.success(f"Found {len(recommendations)} relevant assessments:")
                    
                    # Convert to DataFrame for display
                    df = pd.DataFrame(recommendations)
                    
                    # Rename columns for display
                    df = df.rename(columns={
                        'name': 'Assessment Name',
                        'url': 'URL',
                        'remote_testing': 'Remote Testing',
                        'adaptive_support': 'Adaptive/IRT Support',
                        'duration': 'Duration',
                        'test_type': 'Test Type'
                    })
                    
                    # Format URLs as clickable links
                    df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if x else '')
                    
                    # Display results in an interactive table
                    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Download as CSV option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "shl_recommendations.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    if submit_button:
        st.warning("Please enter a query to get recommendations.")
    else:
        st.info("Enter your requirements above and click 'Get Recommendations'.")

# Footer
st.markdown("---")
st.markdown("Powered by AI and natural language processing")