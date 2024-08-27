import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\saura\OneDrive\Desktop\sharaz\invoice categorisation and tagging\invoice.csv")  # CSV file with 'text' and 'category' columns

# Preprocessing: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['category']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model and vectorizer
joblib.dump(model, 'invoice_categorization_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
