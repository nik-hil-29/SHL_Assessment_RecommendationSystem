import os
import json
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from recommendation_system import SHLRecommendationSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check environment and initialize paths
processed_data_path = "FinalDataSource/processed_assessments.json"
db_directory = "chroma_db"

# Initialize recommendation system
try:
    recommendation_system = SHLRecommendationSystem(
        processed_data_path=processed_data_path,
        db_directory=db_directory
    )
    logger.info("Recommendation system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommendation system: {e}")
    recommendation_system = None

# Define response models
class AssessmentResponse(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_support: str
    duration: str
    test_type: str

class RecommendationResponse(BaseModel):
    recommendations: List[AssessmentResponse]

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions or queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the SHL Assessment Recommendation API"}

@app.get("/health")
async def health_check():
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    return {"status": "healthy"}

@app.get("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    query: str = Query(..., description="The job description or query text"),
    max_results: Optional[int] = Query(10, description="Maximum number of recommendations to return", ge=1, le=10)
):
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    try:
        # Get recommendations as DataFrame
        recommendations_df = recommendation_system.get_recommendations(query, max_results=max_results)
        
        # Convert DataFrame to response model
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendations.append(AssessmentResponse(
                name=row["Assessment Name"],
                url=row["URL"],
                remote_testing=row["Remote Testing"],
                adaptive_support=row["Adaptive/IRT Support"],
                duration=row["Duration"],
                test_type=row["Test Type"]
            ))
        
        return RecommendationResponse(recommendations=recommendations)
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

# Run server directly when script is executed
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)