# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Load dataset
# Replace 'sentiment_data.csv' with your dataset file
df = pd.read_csv(r"C:\Users\saura\OneDrive\Desktop\sharaz\sentiment_data.csv")  # Ensure this CSV has 'text' and 'sentiment' columns

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\saura\OneDrive\Desktop\sharaz\sentiment_data.csv")

# Prepare the data
X = df['text']
y = df['sentiment']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'sentiment_model.joblib')

# Make predictions
y_pred = model.predict(X_test)

# Print classification report with zero_division parameter
report = metrics.classification_report(y_test, y_pred, zero_division=0)
print(report)

# Additional metrics
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print("Accuracy Score:")
print(metrics.accuracy_score(y_test, y_pred))

