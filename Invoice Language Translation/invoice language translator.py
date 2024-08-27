import pytesseract
from deep_translator import GoogleTranslator
from PIL import Image

# Load the image
image_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\language transalotor\Rechnung-Vorlage-Deutsch (1).png"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = Image.open(image_path)

# Use pytesseract to extract text
extracted_text = pytesseract.image_to_string(img, lang='deu')

# Print the extracted text (optional, to see the German text)
print("Extracted German Text:")
print(extracted_text)

# Initialize the GoogleTranslator
translator = GoogleTranslator(source='de', target='en')

# Translate the text to English
translated_text = translator.translate(extracted_text)

# Print the translated text
print("\nTranslated English Text:")
print(translated_text)
