import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import io

# Define colors
default_color = (0, 0, 255)  # Red for text only
numerical_color = (0, 255, 0)  # Green for numerical values only
mixed_color = (255, 0, 0)  # Blue for mixed content


# Function to process PDF and draw rectangles
def process_pdf(uploaded_file, output_path):
    # Open the uploaded PDF file
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    processed_images = []

    # Loop through each page in the PDF
    for i in range(len(doc)):
        # Get the PyMuPDF page
        pdf_page = doc.load_page(i)

        # Extract text boxes
        text_boxes = pdf_page.get_text("dict")["blocks"]

        # Create an image from the page
        pix = pdf_page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))  # Use io.BytesIO to handle image bytes

        # Convert image to NumPy array for OpenCV processing
        img_np = np.array(img)

        # Draw rectangles around each text box
        for block in text_boxes:
            if block['type'] == 0:  # Text block
                for line in block['lines']:
                    for span in line['spans']:
                        x0, y0 = span['bbox'][0], span['bbox'][1]
                        x1, y1 = span['bbox'][2], span['bbox'][3]

                        # Check content types
                        contains_digit = any(char.isdigit() for char in span['text'])
                        contains_letter = any(char.isalpha() for char in span['text'])

                        # Set color based on content
                        if contains_digit and contains_letter:
                            color = mixed_color
                        elif contains_digit:
                            color = numerical_color
                        else:
                            color = default_color

                        # Draw the rectangle with the determined color
                        cv2.rectangle(img_np, (int(x0), int(y0)), (int(x1), int(y1)), color,
                                      2)  # Draw rectangle with determined color

        # Convert NumPy array back to image
        processed_img = Image.fromarray(img_np)
        processed_images.append(processed_img)

        # Save the image with boundaries (optional)
        processed_img.save(f"{output_path}_page_{i + 1}.png")

    return processed_images


# Streamlit UI
st.title("PDF Text Box Highlighter")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    output_path = "output"  # Define the output path for saved images
    processed_images = process_pdf(uploaded_file, output_path)

    st.write("Processed Pages:")
    for i, img in enumerate(processed_images):
        st.image(img, caption=f"Page {i + 1}", use_column_width=True)
