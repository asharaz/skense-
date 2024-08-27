# PDF Text Box Highlighter

## Project Background

The PDF Text Box Highlighter is a tool designed to help users visually parse and analyze PDF documents by highlighting different types of content within text boxes. The application highlights text blocks in PDFs based on whether they contain alphabetical, numerical, or mixed content. This feature is particularly useful in scenarios where understanding the content type quickly is necessary, such as in invoice processing, data extraction, or document review.

The project is implemented using Python and leverages several powerful libraries like `PyMuPDF` (for PDF handling), `OpenCV` (for image processing), and `Streamlit` (for creating a user-friendly web interface).

## Code Explanation

### Importing Libraries
```python
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import io


```
1.streamlit: Used for creating a web interface where users can upload PDFs and view processed results.

2.fitz: Part of the PyMuPDF library, which is used to work with PDF files, extract text, and process pages.

3.PIL: Python Imaging Library, used for opening, manipulating, and saving images.

4.numpy: A powerful library for numerical computing, used here for handling image data as arrays.

5.cv2: OpenCV library, used for drawing rectangles and processing images.

6.io: Used for handling byte streams, allowing conversion between different data formats.

### Defining Colors for Highlighting
```python
default_color = (0, 0, 255)  # Red for text only
numerical_color = (0, 255, 0)  # Green for numerical values only
mixed_color = (255, 0, 0)  # Blue for mixed content
```

These tuples define RGB colors used to highlight different types of content in the PDF:

1.Red for alphabetical content.

2.Green for numerical content.

3.Blue for mixed content.

### Processing PDF and Drawing Rectangles

#### Function Definition

``` python
def process_pdf(uploaded_file, output_path):
```
 * uploaded_file: This parameter represents the PDF file that has been uploaded by the user.
* output_path: This is the base path for saving the processed images of each PDF page.
 
##### Open PDF File
``` python   
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
```
* fitz.open: This function from the PyMuPDF library opens the PDF file.
* stream=uploaded_file.read(): Reads the content of the uploaded PDF file.
* filetype="pdf": Specifies the type of file being opened

##### Initialize List for Processed Images
``` python 
processed_images = []
```
* processed_images: A list to store the processed images of each page.

##### Process Each Page
``` python 
    for i in range(len(doc)):
        pdf_page = doc.load_page(i)
        text_boxes = pdf_page.get_text("dict")["blocks"]
```
* for i in range(len(doc)): Loop through each page of the PDF.
* pdf_page = doc.load_page(i): Loads the i-th page of the PDF.
* text_boxes = pdf_page.get_text("dict")["blocks"]: Extracts text blocks from the page as a dictionary.

##### Convert PDF Page to Image
``` python 
        pix = pdf_page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))

        img_np = np.array(img)
 ```
* pix = pdf_page.get_pixmap(): Converts the page to an image (pixmap).
* img = Image.open(io.BytesIO(pix.tobytes())): Opens the image from bytes using PIL.
* img_np = np.array(img): Converts the image to a NumPy array for processing with OpenCV.

##### Highlight Text Boxes
``` python 
        for block in text_boxes:
            if block['type'] == 0:
                for line in block['lines']:
                    for span in line['spans']:
                        x0, y0 = span['bbox'][0], span['bbox'][1]
                        x1, y1 = span['bbox'][2], span['bbox'][3]

                        contains_digit = any(char.isdigit() for char in span['text'])
                        contains_letter = any(char.isalpha() for char in span['text'])

                        if contains_digit and contains_letter:
                            color = mixed_color
                        elif contains_digit:
                            color = numerical_color
                        else:
                            color = default_color

                        cv2.rectangle(img_np, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
```
* for block in text_boxes: Iterates over each block of text.
* if block['type'] == 0: Checks if the block is a text block.
* for line in block['lines']: Iterates over each line in the block.
* for span in line['spans']: Iterates over each span (segment) of text in the line.
* x0, y0, x1, y1: Coordinates for the bounding box of the text span.
* contains_digit: Checks if the text contains any digits.
* contains_letter: Checks if the text contains any alphabetical characters.
* color: Determines the color for highlighting based on whether the text contains digits, letters, or both.
* cv2.rectangle(img_np, (int(x0), int(y0)), (int(x1), int(y1)), color, 2): Draws a rectangle around the text span on the image with the determined color.

##### Save Processed Image
``` python
        processed_img = Image.fromarray(img_np)
        processed_images.append(processed_img)
        processed_img.save(f"{output_path}_page_{i + 1}.png")
```
* processed_img = Image.fromarray(img_np): Converts the NumPy array back to a PIL image.
* processed_images.append(processed_img): Adds the processed image to the list.
* processed_img.save(f"{output_path}_page_{i + 1}.png"): Saves the processed image to a file with a name indicating the page number.

##### Return Processed Images

``` python
    return processed_images
 ```
* return processed_images: Returns the list of processed images
    
### Streamlit User Interface

##### Streamlit Interface Setup
``` python
st.title("PDF Text Box Highlighter")
```
* st.title("PDF Text Box Highlighter"): Sets the title of the web application. This will appear at the top of the Streamlit app as a heading.

##### File Upload Widget
``` python
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
```
* st.file_uploader("Upload a PDF file", type=["pdf"]): Creates a file uploader widget in the web app that allows users to upload files.

* "Upload a PDF file": Label for the file uploader.
* type=["pdf"]: Restricts the file types to PDFs only.
* uploaded_file: This variable holds the file object of the uploaded PDF. If a user uploads a file, this variable will contain the PDF file; otherwise, it will be None.
##### Conditional Processing
```python
if uploaded_file is not None:
    output_path = "output"
    processed_images = process_pdf(uploaded_file, output_path)
```
* if uploaded_file is not None:: Checks if a file has been uploaded.
If uploaded_file is not None (meaning a file has been uploaded), the code inside this block will execute.
* output_path = "output": Defines the base path where the processed images will be saved. The actual file names will include page numbers.
* processed_images = process_pdf(uploaded_file, output_path): Calls the process_pdf function with the uploaded PDF file and the output path. This function processes the PDF and returns a list of images with highlighted text boxes.
    
##### Displaying Processed Images
```python
st.write("Processed Pages:")
    for i, img in enumerate(processed_images):
        st.image(img, caption=f"Page {i + 1}", use_column_width=True)
```
* st.write("Processed Pages:"): Displays a text label on the app indicating that the processed pages will follow.

* for i, img in enumerate(processed_images):: Loops through the list of processed images.
  * i: The index of the image (corresponding to the page number).
  * img: The processed image.
* st.image(img, caption=f"Page {i + 1}", use_column_width=True): Displays each image in the Streamlit app.
  * img: The image to be displayed.
  * caption=f"Page {i + 1}": Adds a caption indicating the page number.
  * use_column_width=True: Ensures that the image width adjusts to fit the column width of the app.

## Output
Run the code,as soon as it runs you will have to run a streamlit command on command prompt.
![Screenshot 2024-08-26 112626](https://github.com/user-attachments/assets/d4e2c9ed-1597-4ea2-baf8-e75a3b29aee4)



Copy the above command and run in the command prompt which will create a link on the url.
![Screenshot 2024-08-26 114334](https://github.com/user-attachments/assets/db0df012-a618-437a-b962-f790404a3041)



Now copy the url and paste it on your browser you will be redirected to the streamlit ui.
![Screenshot 2024-08-26 114406](https://github.com/user-attachments/assets/ea6797da-d138-4232-a00d-abf8a0969967)



Here you can browse the file that you need to be bounded ,once you select the file it shows you the output.
![1](https://github.com/user-attachments/assets/c5cb9ea8-5c53-4674-b73c-c317876fc7f2)

