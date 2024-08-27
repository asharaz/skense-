# Image Text Extraction and Translation
## Project Overview
This project demonstrates how to extract text from an image using Optical Character Recognition (OCR) with Tesseract, and then translate the extracted text from German to English using the deep_translator library. This can be particularly useful in scenarios where you have documents, invoices, or any text in an image format in a foreign language that needs to be translated for better understanding or processing.

The script performs the following tasks:

* Text Extraction: Extracts text from an image using the Tesseract-OCR engine.
* Text Translation: Translates the extracted text from German to English using Google Translate via the deep_translator library.
* Output: Displays both the extracted German text and the translated English text.
## Prerequisites

You will need the following Python libraries:

1. pytesseract: A Python wrapper for the Tesseract-OCR engine.
2. Pillow: A Python Imaging Library (PIL) fork, used for opening and manipulating images.
3. deep-translator: A simple translation package that allows you to translate text using various translation engines.
4. Tesseract-OCR: You need to install Tesseract-OCR on your system.
## Code Explanation
### Importing Required Libraries
```python
import pytesseract
from deep_translator import GoogleTranslator
from PIL import Image
```
* pytesseract: A Python wrapper for Google's Tesseract-OCR Engine, used for text extraction from images.
* GoogleTranslator: A class from the deep_translator library that allows translating text between different languages using Google Translate.
* Image: A class from the Pillow library, which is used for opening and manipulating images.
### Loading the Image
```python
image_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\language transalotor\Rechnung-Vorlage-Deutsch (1).png"
img = Image.open(image_path)
```
* image_path: The path to the image file containing German text. Update this path with the actual location of your image file.
* Image.open(image_path): Opens the image file specified by the image_path.
### Setting Tesseract-OCR Path and Extracting Text
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
extracted_text = pytesseract.image_to_string(img, lang='deu')
```
* pytesseract.pytesseract.tesseract_cmd: Specifies the path to the Tesseract-OCR executable on your system.
* image_to_string(img, lang='deu'): Extracts text from the opened image img. The lang='deu' parameter indicates that the text is in German, so Tesseract will use the German language model for OCR.
### Printing the Extracted Text (Optional)
```python
print("Extracted German Text:")
print(extracted_text)
```
* print("Extracted German Text:"): Prints a label indicating that the following text is the extracted German text.
* print(extracted_text): Outputs the extracted text to the console.
### Translating the Extracted Text
``` python
translator = GoogleTranslator(source='de', target='en')
translated_text = translator.translate(extracted_text)
```
* GoogleTranslator(source='de', target='en'): Initializes the GoogleTranslator object, specifying German ('de') as the source language and English ('en') as the target language.
* translator.translate(extracted_text): Translates the extracted German text into English.
### Printing the Translated Text
```python
print("\nTranslated English Text:")
print(translated_text)
```
* print("\nTranslated English Text:"): Prints a label indicating that the following text is the translated English text.
* print(translated_text): Outputs the translated text to the console.
### Running the Script
1. Make sure the image file path is correct in the image_path variable.
2. Ensure Tesseract-OCR is installed and the path to the executable is correctly set in the script.

## Output 
Before running the code make sure the file path is changed accordingly.the file path added here takes to images as shown.
![Rechnung-Vorlage-Deutsch (1)](https://github.com/user-attachments/assets/16d3be45-d745-4939-9703-d6d03427cb3d)


As soon as you the run code,the code extracts all the text from the file.
![Screenshot 2024-08-26 135515](https://github.com/user-attachments/assets/343dcb31-c5b6-4d13-a613-b499f2c73c3e)


It then translates the words to common known language in english.
![Screenshot 2024-08-26 135713](https://github.com/user-attachments/assets/94c310d4-4487-4122-b012-c05dfd2b925b)



