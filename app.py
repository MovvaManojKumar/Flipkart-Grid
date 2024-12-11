import streamlit as st
import os
import torch
from paddleocr import PaddleOCR
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from datetime import datetime, timedelta
import re
from ultralytics import YOLO
import pandas as pd

# Initialize models
ocr = PaddleOCR(lang='en')
fruit_model = load_model('DenseNet20_model.h5')
brand_model = YOLO('b.pt')

# Class names for fruit freshness classification
class_names = {
    0: 'Banana_Bad',
    1: 'Banana_Good',
    2: 'Fresh',
    3: 'FreshCarrot',
    4: 'FreshCucumber',
    5: 'FreshMango',
    6: 'FreshTomato',
    7: 'Guava_Bad',
    8: 'Guava_Good',
    9: 'Lime_Bad',
    10: 'Lime_Good',
    11: 'Rotten',
    12: 'RottenCarrot',
    13: 'RottenCucumber',
    14: 'RottenMango',
    15: 'RottenTomato',
    16: 'freshBread',
    17: 'rottenBread'
}

# Helper functions
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def extract_expiry_dates(text):
    expiry_date_patterns = [
        r'USE BY (\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Use by date
        r'BEST BEFORE (\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Best before date
        r'EXPIRY DATE[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiry Date: 20/07/2024
        r'EXPIRY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiry: 20/07/2024
        r'EXP[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Exp: 20/07/2024
        r'VALID UNTIL[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Valid Until: 20/07/2024
        r'CONSUME BY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Consume By: 20/07/2024
        r'USE BY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Use By: 20/07/2024
        r'BEST BEFORE[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Best Before: 20/07/2024
        r'EXPIRES ON[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expires On: 20/07/2024
        r'EXPIRES[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expires: 20/07/2024
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # General date format
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',  # Short year date format
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})',  # Date with month name and full year
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{2})',  # Date with month name and short year
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',  # Year-month-day format
        r'(\d{4}[A-Za-z]{3,}\d{1,2})',  # Year with month name
        r'(\d{1,2}[A-Za-z]{3,}\d{4})',  # Day with month name and full year
        r'Best before (\d+) months',  # Best before in months
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Expiry Date: 20/07/2024
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Expiry Date: 20 MAY 2024
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 2024/07/2O24
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Expiry Date: 2024/07/20
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}))',  # Best Before: 2025
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Best Before: 20/07/2024
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
        r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
        r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Best Before: 20 MAY 2024
        r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 2024/07/2O24
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Best Before: 2024/07/20
        r'(?:best\s*before\s*[:\-]?\s*.*?(\d{1,2}\d{2}\d{2}))',  # Best Before: 20250720
        r'(?:best\s*before\s*[:\-]?\s*(\d{6}))',  # Best Before: 202507
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}[0O]\d{2}))',  # Consume Before: 3ODEC2O24
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}\d{2}))',  # Consume Before: 30DEC23
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Consume Before: 20/07/2024
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
        r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
        r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Consume Before: 20 MAY 2024
        r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 2024/07/2O24
        r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Consume Before: 2024/07/20
        r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
        r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Exp: 20/07/2024
        r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
        r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
        r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Exp: 20 MAY 2024
        r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
        r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 2024/07/2O24
        r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Exp: 2024/07/20
        r"Exp\.Date\s+(\d{2}[A-Z]{3}\d{4})",  # Expiration date format
        r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 16 MAR 2O30 (with typo)
        r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp. Date: 15/12/2O30 (with typo)
        r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 15 MAR 2O30 (with typo)
        r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date cdsyubfuyef 15 MAR 2O30 (with typo)
        r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # 20/07/2024
        r'(\d{2}[\/\-]\d{2}[\/\-]\d{2})',  # 20/07/24
        r'(\d{2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
        r'(\d{2}\s*[A-Za-z]{3,}\s*\d{2})',  # 20 MAY 24
        r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024/07/20
        r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024-07-20
        r'(\d{4}[A-Za-z]{3,}\d{2})',  # 2024MAY20
        r'(\d{2}[A-Za-z]{3,}\d{4})',  # 20MAY2024
        r'(?:DX3\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # DX3 date format
        r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*[A-Za-z]{3,}\s*(\d{4}|\d{2})))',  # Exp. Date format
        r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # Exp. Date: 20 05 2025
        r'(\d{4}[A-Za-z]{3}\d{2})',  # 2025MAY11
        r'(?:best\s*before\s*[:\-]?\s*(\d+)\s*(days?|months?|years?))',  # Best Before: 6 months
        r'(?:best\s*before\s*[:\-]?\s*(three)\s*(months?))',  # Best Before: three months
        r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*\d{4})',  # Month name with year
        r'\bUSE BY\s+(\d{1,2}[A-Za-z]{3}\d{4})\b',  # Use by with month name
        r"Exp\.Date\s*(\d{2}[A-Z]{3}\d{4})",  # Expiration date format
        r"EXP:\d{4}/\d{2}/\d{4}/\d{1}/[A-Z]"  # EXP format
    ]
    
    current_date = datetime.now()
    dates_info = []

    for pattern in expiry_date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Try parsing the date
                expiry_date = datetime.strptime(date_str, '%d/%m/%Y')
            except ValueError:
                try:
                    expiry_date = datetime.strptime(date_str, '%d-%m-%Y')
                except ValueError:
                    # Add more date formats as needed
                    continue

            days_left = (expiry_date - current_date).days
            if days_left < 0:
                dates_info.append((date_str, "Expired"))
            else:
                dates_info.append((date_str, f"{days_left} days left"))
            break  # Stop after finding the first valid date

    return dates_info

# Streamlit app
st.title("Image Processing Application")

# User choice for processing
task_choice = st.radio("Choose a task", ("Text and Brand Detection", "Freshness Detection", "Object Detection"))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure the uploads directory exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # Save the uploaded file
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(image_path, caption='Uploaded Image', use_container_width=True)

    if task_choice == "Text and Brand Detection":
        # Text Extraction
        st.header("Text Extraction")
        result = ocr.ocr(image_path)
        text = ' '.join([line[1][0] for line in result[0]])
        st.write("Extracted Text:")
        st.text(text)

        # Expiry Date Extraction
        st.header("Expiry Date Extraction")
        expiry_dates_info = extract_expiry_dates(text)
        if expiry_dates_info:
            st.write("Expiry Dates Found:")
            for date_str, days_left in expiry_dates_info:
                st.text(f"{date_str} - {days_left}")
        else:
            st.write("No expiry dates found.")

        # Brand Prediction
        st.header("Brand Prediction")
        results = brand_model(image_path)
        detected_brands = []
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                class_id = int(box[5])
                confidence = box[4]  # Assuming the confidence score is at index 4
                detected_brands.append((result.names[class_id], confidence))
        
        # Count occurrences and average confidence of each brand
        brand_data = {}
        for brand, confidence in detected_brands:
            if brand in brand_data:
                brand_data[brand]['count'] += 1
                brand_data[brand]['total_confidence'] += confidence
            else:
                brand_data[brand] = {'count': 1, 'total_confidence': confidence}
        
        # Prepare data for display
        brand_display_data = [
            {'Object': brand, 'Count': data['count'], 'Average Confidence': data['total_confidence'] / data['count']}
            for brand, data in brand_data.items()
        ]
        
        # Display detected brands in a table with column names
        st.write("Detected Brands and Counts:")
        st.table(pd.DataFrame(brand_display_data))

    elif task_choice == "Freshness Detection":
        # Freshness Prediction
        st.header("Freshness Prediction")
        img_array = preprocess_image(image_path)
        predictions = fruit_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        st.write("Predicted Freshness:", class_names[predicted_class])

    elif task_choice == "Object Detection":
        # Object Detection
        st.header("Object Detection")
        results = brand_model(image_path)  # Using YOLOv5s model for detection
        detected_objects = []
        
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                class_id = int(box[5])
                confidence = box[4]  # Assuming the confidence score is at index 4
                detected_objects.append((result.names[class_id], confidence))
        
        # Count occurrences and average confidence of each object
        object_data = {}
        for obj, confidence in detected_objects:
            if obj in object_data:
                object_data[obj]['count'] += 1
                object_data[obj]['total_confidence'] += confidence
            else:
                object_data[obj] = {'count': 1, 'total_confidence': confidence}
        
        # Prepare data for display
        object_display_data = [
            {'Object': obj, 'Count': data['count'], 'Average Confidence': data['total_confidence'] / data['count']}
            for obj, data in object_data.items()
        ]
        
        # Display detected objects in a table with column names
        st.write("Detected Objects and Counts:")
        st.table(pd.DataFrame(object_display_data))