from fastapi import APIRouter
from app.schemas import MicrosoftInput,MicrosoftOutput
from predict import make_prediction

api_router = APIRouter()

@api_router.get("/health")
def health():
  return {"status":"ok"}

@api_router.post("/predict",response_model=MicrosoftOutput)
def predict(input_data:MicrosoftInput):
    result = make_prediction(input_data.dict())
    return result