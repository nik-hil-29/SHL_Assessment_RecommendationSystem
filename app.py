# app.py
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
import os
import json
from recommendation_system import SHLRecommendationSystem

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

# Check environment and initialize paths
processed_data_path = "FinalDataSource/processed_assessments.json"
db_directory = "chroma_db"

# Initialize recommendation system
recommendation_system = SHLRecommendationSystem(
    processed_data_path=processed_data_path,
    db_directory=db_directory
)

@app.get("/")
async def root():
    return {"message": "Welcome to the SHL Assessment Recommendation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    query: str = Query(..., description="The job description or query text"),
    max_results: Optional[int] = Query(10, description="Maximum number of recommendations to return", ge=1, le=10)
):
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
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
