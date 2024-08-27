# Sentiment Analysis API
## Project Overview
This project involves building a sentiment analysis model using a Naive Bayes classifier and deploying it via a FastAPI web service. The project consists of two main components:

1. Model Training Script (train_model.py): This script processes text data, trains a Naive Bayes model using the data, and saves the trained model for later use.
2. API Service (main.py): This script sets up a FastAPI server that serves the sentiment analysis model. Users can send text inputs to the API, and it will return the predicted sentiment.

## Prerequisites
1. pandas: For data manipulation and loading CSV files.
2. scikit-learn: For building and evaluating the sentiment analysis model.
3. joblib: For saving and loading the trained model.
4. fastapi: For creating the web API.
5. pydantic: For validating and parsing the API request data.
6. uvicorn: For running the FastAPI application.

## Code Explanation for model

### Importing Libraries
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
```
* pandas: For data manipulation and analysis.
* CountVectorizer: Converts a collection of text documents into a matrix of token counts.
* MultinomialNB: A Naive Bayes classifier suitable for classification with discrete features.
* make_pipeline: Combines multiple steps (like feature extraction and model training) into a single pipeline.
* train_test_split: Splits the dataset into training and testing subsets.
* metrics: Provides functions to evaluate the model's performance.
* joblib: For saving and loading the trained model. 
### Loading the Dataset
```python
df = pd.read_csv(r"C:\Users\saura\OneDrive\Desktop\sharaz\sentiment_data.csv")
```
* pd.read_csv(): Reads a CSV file into a DataFrame. This CSV contains text data and their corresponding sentiments. 
### Preparing the Data
```python
X = df['text']
y = df['sentiment']
```
* X: The input features (text) for the model.
* y: The target variable (sentiment) for the model. 
### Splitting the Data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
* train_test_split(): Divides the data into training (80%) and testing (20%) subsets. The random_state ensures reproducibility.
### Creating and Training the Model
```python
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
```
* make_pipeline(): Constructs a pipeline that first transforms the text data into a matrix of token counts using CountVectorizer and then applies the Naive Bayes classifier (MultinomialNB).
* model.fit(X_train, y_train): Trains the model using the training data. 
### Saving the Trained Model
```python
joblib.dump(model, 'sentiment_model.joblib')
```
* joblib.dump(): Saves the trained model to a file named sentiment_model.joblib for future use. 
### Making Predictions
```python
y_pred = model.predict(X_test)
```
* model.predict(): Uses the trained model to predict sentiments for the test data.
### Evaluating the Model
```python
report = metrics.classification_report(y_test, y_pred, zero_division=0)
print(report)
```
* metrics.classification_report(): Generates a report displaying key classification metrics (e.g., precision, recall, f1-score) for each sentiment class.
* print(report): Outputs the classification report.
### Additional Evaluation Metrics
``` python

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
```
* metrics.confusion_matrix(): Generates a confusion matrix to display the counts of correct and incorrect predictions for each class.
* print(): Outputs the confusion matrix.
### Accuracy score
```python
print("Accuracy Score:")
print(metrics.accuracy_score(y_test, y_pred))
```
* metrics.accuracy_score(): Computes and displays the overall accuracy of the model on the test data.

## Code Explanation for sentiment analysis
### Importing Libraries
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
```
* FastAPI: A modern web framework for building APIs with Python.
* HTTPException: Handles HTTP errors and exceptions.
* BaseModel: Pydantic’s model for data validation using Python type annotations.
* joblib: Loads the trained model for use in predictions. 
### Initializing the FastAPI Application
```python
app = FastAPI()
```
* FastAPI(): Creates a new FastAPI application instance. 
### Loading the Trained Model
```python
model = joblib.load('sentiment_model.joblib')
```
* joblib.load(): Loads the pre-trained sentiment analysis model from the sentiment_model.joblib file. 
### Defining the Input Schema
``` python
class FeedbackRequest(BaseModel):
    text: str
```
* FeedbackRequest(BaseModel): A Pydantic model that defines the expected input format for the API. It requires a single field, text.
### Root Endpoint
```python
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}
```
* @app.get("/"): Defines a GET request handler for the root URL (/).
* async def root(): An asynchronous function that returns a welcome message.
* return {"message": "Welcome to the Sentiment Analysis API!"}: Sends a JSON response containing a welcome message.
### Sentiment Analysis Endpoint
```python
@app.post("/analyze")
async def analyze_sentiment(request: FeedbackRequest):
    try:
        text = request.text
        prediction = model.predict([text])
        sentiment = prediction[0]
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
* app.post("/analyze"): Defines a POST request handler at the /analyze endpoint.
* async def analyze_sentiment(request: FeedbackRequest): Handles the sentiment analysis request.
* text = request.text: Extracts the text field from the request.
* model.predict([text]): Uses the loaded model to predict the sentiment of the input text.
* sentiment = prediction[0]: Retrieves the predicted sentiment from the model’s output.
* return {"sentiment": sentiment}: Returns the predicted sentiment as a JSON response.
* except Exception as e: Catches any errors that occur during prediction.
* raise HTTPException(status_code=500, detail=str(e)): Returns an HTTP 500 error if an exception occurs.
### Running the Application
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```
* if name == "main":: Ensures the script is being run directly, not imported as a module.
* import uvicorn: Imports Uvicorn, a fast ASGI server for serving FastAPI applications.
* uvicorn.run(app, host="127.0.0.1", port=8000): Starts the FastAPI application on host 127.0.0.1 and port 8000.

## Output 

Initially check the csv file created with text and sentiments and run the model code to create the model.


![Screenshot 2024-08-27 124601](https://github.com/user-attachments/assets/dc89abde-eea6-43bb-b85a-d7b1bb8f00b1)


When you run the main code,its provides you with the url http://127.0.0.1:8000,now copy this url and paste on the browser.
![Screenshot 2024-08-27 124635](https://github.com/user-attachments/assets/08fc7108-bf22-4935-9174-53f2214197cf)


when you run on the browser, you will get a page saying the message.
![Screenshot 2024-08-27 124655](https://github.com/user-attachments/assets/7c229324-f3ad-498f-a3c4-5d306fde1b0b)


Now open the page with url http://127.0.0.1:8000/docs ,this to open the fast api page.
![Screenshot 2024-08-27 124718](https://github.com/user-attachments/assets/9b788050-6c5c-4484-b4bd-a119e11f1bb6)


In the fast API page,click on post dropdown,where you would be able to writen the sentiment.
![Screenshot 2024-08-27 124902](https://github.com/user-attachments/assets/9f7ac928-80ca-46ee-a9f9-96da6d66a8ac)


After Writing the sentiment scroll down and click the execute button to the see the output result.
![Screenshot 2024-08-27 130232](https://github.com/user-attachments/assets/42960de1-9d15-4fc4-9435-2d1f824f29b8)

