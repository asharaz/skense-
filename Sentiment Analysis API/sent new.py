# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the pre-trained model
model = joblib.load('sentiment_model.joblib')

# Define the input schema
class FeedbackRequest(BaseModel):
    text: str

# Define the root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Define the sentiment analysis endpoint
@app.post("/analyze")
async def analyze_sentiment(request: FeedbackRequest):
    try:
        text = request.text
        prediction = model.predict([text])
        sentiment = prediction[0]
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
