# Invoice Categorization and OCR Processing

## Project Overview
This project utilizes a Logistic Regression model to categorize invoice data into predefined categories. Additionally, it leverages the Tesseract OCR engine to extract text from images of invoices, which is then categorized using the trained model.

## Prerequisites
Before running the code, make sure you have the following installed:

* Python 3.x

* pip (Python package installer)

* Required Python packages:

* pandas

* scikit-learn

* joblib

* pytesseract

* Pillow

* Additionally, you need to have Tesseract OCR installed on your system.

## Code Explanation

### Importing Libraries for the model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

```
* pandas as pd: Imports the pandas library, which is used for data manipulation and analysis, and is given the alias pd.
* sklearn.model_selection.train_test_split: Imports the train_test_split function from scikit-learn to split the dataset into training and test sets.
* sklearn.feature_extraction.text.TfidfVectorizer: Imports the TfidfVectorizer, which converts a collection of raw documents into a matrix of TF-IDF features.
* sklearn.linear_model.LogisticRegression: Imports the LogisticRegression model, which is used for classification tasks.
* sklearn.metrics.classification_report, confusion_matrix: Imports tools for evaluating the performance of a classification model.
* joblib: Imports the joblib library, which is used to save and load Python objects (e.g., models).
### Loading the Dataset
```python
data = pd.read_csv(r"C:\Users\saura\OneDrive\Desktop\sharaz\invoice.csv")  # CSV file with 'text' and 'category' columns
```
* data = pd.read_csv(...): Reads the CSV file containing the invoice data into a pandas DataFrame. The file is expected to have two columns: 'text' and 'category'.

### Preprocessing: Convert Text to Numerical Features using TF-IDF
```python
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['category']
```
* TfidfVectorizer(max_features=5000): Initializes the TF-IDF vectorizer with a maximum of 5000 features.
* X = vectorizer.fit_transform(data['text']): Fits the vectorizer on the 'text' column of the DataFrame and transforms the text into numerical features. The result is stored in X.
* y = data['category']: Extracts the 'category' column from the DataFrame and stores it in y.
### Splitting Data into Training and Test Sets
```python

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
* train_test_split(X, y, test_size=0.2, random_state=42): Splits the data into training and test sets, with 80% of the data used for training and 20% for testing. The random_state=42 ensures reproducibility.
### Training a Logistic Regression Model
```python

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
* LogisticRegression(max_iter=1000): Initializes a Logistic Regression model with a maximum of 1000 iterations.
* model.fit(X_train, y_train): Trains the model using the training data (X_train and y_train).
* Evaluating the Model on the Test Set
```python

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
```
* y_pred = model.predict(X_test): Uses the trained model to make predictions on the test set.
* print(classification_report(y_test, y_pred, zero_division=0)): Prints a classification report, showing precision, recall, F1-score, and support for each class. zero_division=0 prevents division by zero errors.
### Saving the Trained Model and Vectorizer
```python
joblib.dump(model, 'invoice_categorization_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```
* joblib.dump(model, 'invoice_categorization_model.pkl'): Saves the trained model to a file named invoice_categorization_model.pkl.
* joblib.dump(vectorizer, 'tfidf_vectorizer.pkl'): Saves the fitted vectorizer to a file named tfidf_vectorizer.pkl.

### Importing Necessary Libraries for OCR
```python
import pytesseract
from PIL import Image
import joblib
```
* pytesseract: Imports the pytesseract library, which is a Python wrapper for Google’s Tesseract-OCR Engine.
* PIL.Image: Imports the Image module from the Pillow library, used for opening and manipulating image files.
* joblib: Re-imports the joblib library for loading the saved model and vectorizer.
### Setting Path to Tesseract Executable
```python

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
* pytesseract.pytesseract.tesseract_cmd: Sets the path to the Tesseract executable on your system, allowing pytesseract to run OCR on images.
### Loading the Trained Model and Vectorizer
```python
model = joblib.load('invoice_categorization_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
```
* model = joblib.load('invoice_categorization_model.pkl'): Loads the trained model from the saved file.
* vectorizer = joblib.load('tfidf_vectorizer.pkl'): Loads the fitted TF-IDF vectorizer from the saved file.
### Defining Functions for OCR and Categorization
```python
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text
```
* extract_text_from_image(image_path): Defines a function to open an image file and extract text from it using Tesseract OCR.
* Image.open(image_path): Opens the image from the specified file path.
* pytesseract.image_to_string(image): Uses Tesseract to extract text from the image.
###  Returns the extracted text.
```python
def preprocess_text(text):
    """Preprocess the extracted text."""
    text = text.lower().strip()
    return text
```
* preprocess_text(text): Defines a function to preprocess the extracted text by converting it to lowercase and stripping whitespace.
* text.lower().strip(): Converts the text to lowercase and removes leading/trailing whitespace.
### Returns the preprocessed text using the trained model.
```python
def categorize_text(text):
    X_new = vectorizer.transform([text])
    category_prediction = model.predict(X_new)
    return category_prediction[0]
 ```
* categorize_text(text): Defines a function to categorize the preprocessed text using the trained model.
* vectorizer.transform([text]): Transforms the input text into numerical features using the TF-IDF vectorizer.
* category_prediction = model.predict(X_new): Uses the trained model to predict the category of the text.
* return category_prediction[0]: Returns the predicted category.
### Main Function to Process the Image and Categorize the Text
```python
def main(image_path):
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text:\n", extracted_text)
    processed_text = preprocess_text(extracted_text)
    category = categorize_text(processed_text)
    print("Predicted Category:", category)
```
* main(image_path): Defines the main function to extract text from an image, preprocess it, categorize it, and print the results.
* extracted_text = extract_text_from_image(image_path): Extracts text from the specified image.
* print("Extracted Text:\n", extracted_text): Prints the extracted text.
* processed_text = preprocess_text(extracted_text): Preprocesses the extracted text.
* category = categorize_text(processed_text): Categorizes the preprocessed text.
* print("Predicted Category:", category): Prints the predicted category.
### Running the Script
```python

if __name__ == "__main__":
    image_path = r"C:\Users\saura\Downloads\medical-store-cash-memo.jpg"
    main(image_path)
```
* if name == "main":: Ensures that the script runs only when executed directly, not when imported as a module.
* image_path = ...: Sets the path to the image file to be processed.
* main(image_path): Calls the main function to process the image and categorize the text.

## Output
Initially create a csv file with text and category in order to create a model.
![img_13.png](img_13.png)

Add the path of the file that you need to predict the category.here is the image of the invoice i would like to predict.
![img_14.png](img_14.png)

Now run the code,the code will predict the which category does it belongs to.
![img_15.png](img_15.png)
