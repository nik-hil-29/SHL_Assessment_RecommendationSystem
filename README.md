# SHL Assessment RecommendationSystem

## Overview

The SHL Assessment Recommendation System is an intelligent AI-powered tool designed to help hiring managers and recruiters find the most appropriate assessments for their specific job requirements. Leveraging advanced natural language processing and vector search technologies, the system provides tailored recommendations from SHL's extensive assessment catalog.

## Key Features

- 🤖 **AI-Powered Recommendations**: Uses Google's Gemini AI to understand job requirements and match them with relevant assessments
- 🔍 **Semantic Search**: Employs vector embeddings for intelligent, context-aware assessment matching
- ⏱️ **Duration Filtering**: Can filter assessments based on time constraints
- 📊 **Multiple Assessment Types**: Supports various assessment categories like aptitude, personality, skills, and more
- 🌐 **Web Interface**: Includes both API and Streamlit web application for easy interaction

## Technology Stack
- **Data Source**-
  - AgentQL(LLM based crawling agent)
  - Playwright(headless)
- **Backend**: 
  - FastAPI
  - Python
- **AI Integration**:
  - Google Gemini
  - Langchain
- **Vector Database**:
  - ChromaDB
- **Web Interface**:
  - Streamlit
- **Deployment**:
  - Docker
  - Docker Compose

## Prerequisites

- Python 3.9+
- Google API Key (for Gemini)
- Docker (optional, for containerized deployment)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SHL_Assessment_RecommendationSystem.git
cd SHL_Assessment_RecommendationSystem
```

### 2. Set Up Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
./setup.sh
```

### 3. Configure API Keys
Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
AGENTQL_API_KEY=your_agentql_api_key_here
```

### 4. Run the Application

#### Local Deployment
```bash
# Start the application
./deploy.sh
```

#### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Usage

### Web Interface
- Navigate to `http://localhost:8501`
- Enter a job description or requirements
- Get instant assessment recommendations

### API Endpoint
- Endpoint: `http://localhost:8000/recommend`
- Parameters:
  - `query`: Job description or requirements
  - `max_results`: Maximum number of recommendations (optional, default 10)

### Example Queries
- "I need a Java programming assessment that can be completed in 30 minutes"
- "Looking for a personality assessment for managerial positions"
- "Need a SQL database test for experienced developers"

## Evaluation

The system includes comprehensive evaluation scripts:

```bash
# Run full evaluation
./run_evaluation.sh
```

Metrics include:
- Mean Recall@K
- Mean Average Precision@K
The evaluation results can be seen in evaluation module

## Project Structure

- `api_server.py`: FastAPI server
- `recommendation_system.py`: Core recommendation logic
- `vector_store.py`: Vector database management
- `gemini_integration.py`: AI integration and query processing
- `streamlit_app.py`: Web interface
- `evaluate.py`: System performance evaluation
- `deploy.sh`: Deployment script
- `docker-compose.yml`: Container orchestration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Maintainer: Nikhil Kushwha
- GitHub: nik-hil-29
- Email: imnikhilkushwaha@gmail.com
  
## Wanna Try
- Streamlit Web-App Link : https://appapppy-v66hmrxddd5hmp35fry7fe.streamlit.app/
- Get API For JSON Query : https://shl-assessment-recommendationsystem.onrender.com/recommend?query=Java%20programming%20test%20under%2030%20minutes&max_results=5


## Acknowledgments

- SHL for their comprehensive assessment catalog
- Google for Gemini AI
- Open-source community for incredible libraries and tools
