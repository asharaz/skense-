import pytesseract
from PIL import Image
import joblib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the trained model and vectorizer
model = joblib.load('invoice_categorization_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def extract_text_from_image(image_path):
    """Extract text from the image using OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


def preprocess_text(text):
    """Preprocess the extracted text."""
    # Example preprocessing: convert to lowercase and strip whitespace
    text = text.lower().strip()
    return text


def categorize_text(text):
    """Categorize the preprocessed text using the trained model."""
    # Transform the text using the vectorizer
    X_new = vectorizer.transform([text])
    # Predict the category
    category_prediction = model.predict(X_new)
    return category_prediction[0]


def main(image_path):
    """Main function to process the image and categorize the text."""
    # Extract text from the image
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text:\n", extracted_text)

    # Preprocess the extracted text
    processed_text = preprocess_text(extracted_text)

    # Categorize the preprocessed text
    category = categorize_text(processed_text)
    print("Predicted Category:", category)


if __name__ == "__main__":
    # Path to the image file (replace with your own file path)
    image_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\invoices\medical-store-cash-memo.jpg"
    main(image_path)
