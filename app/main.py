from fastapi import FastAPI
from app.api import api_router

app = FastAPI(
    title="Microsoft Analyst Recommendation API",
    description="Predicts if analysts will be Bullish or Not Bullish on Microsoft",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)