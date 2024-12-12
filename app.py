import streamlit as st
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import re
from datetime import datetime
import pandas as pd

# Load YOLO models
brand_model = YOLO('b.pt')  # Replace 'b.pt' with the correct path to your YOLO model for brands
ocr = PaddleOCR(lang='en')  # Initialize PaddleOCR
fruit_model = load_model('DenseNet20_model.h5')  # Replace with the correct path to your fruit freshness model
object_model = YOLO('yolov5s.pt')  # Add object detection model

# Class names for freshness detection
class_names = {
    0: 'Banana_Bad', 1: 'Banana_Good', 2: 'Fresh', 3: 'FreshCarrot', 4: 'FreshCucumber',
    5: 'FreshMango', 6: 'FreshTomato', 7: 'Guava_Bad', 8: 'Guava_Good', 9: 'Lime_Bad',
    10: 'Lime_Good', 11: 'Rotten', 12: 'RottenCarrot', 13: 'RottenCucumber',
    14: 'RottenMango', 15: 'RottenTomato', 16: 'freshBread', 17: 'rottenBread'
}

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        margin: 10px 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar {
        background-color: #2c3e50;
        color: white;
    }
    .stSidebar .stSelectbox {
        color: black;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function: Extract expiry dates
def extract_expiry_dates(text):
    patterns = [
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',       # 20/07/2024 or 20-07-2024
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',       # 20/07/24 or 20-07-24
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})',       # 20 MAY 2024
        r'([A-Za-z]{3,}\s*\d{1,2}[,\s]*\d{4})',    # July 20, 2024
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',       # 2024/07/20 or 2024-07-20
        r'([A-Za-z]{3}[\-]\d{1,2}[\-]\d{4})',
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
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{1,2}\d{2}\d{2}))', 
    r'(?:best\s*before\s*[:\-]?\s*(\d{6}))',
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
    r'Exp\.Date\s+(\d{2}[A-Z]{3}\d{4})',
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
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*[A-Za-z]{3,}\s*(\d{4}|\d{2})))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # Exp. Date: 20 05 2025
    r'(\d{4}[A-Za-z]{3}\d{2})',  # 2025MAY11
    r'(?:best\s*before\s*[:\-]?\s*(\d+)\s*(days?|months?|years?))',  # Best Before: 6 months
    r'(?:best\s*before\s*[:\-]?\s*(three)\s*(months?))',
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*\d{4})',
    r'\bUSE BY\s+(\d{1,2}[A-Za-z]{3}\d{4})\b',
    r'Exp\.Date\s*(\d{2}[A-Z]{3}\d{4})',
    r'EXP:\d{4}/\d{2}/\d{4}/\d{1}/[A-Z]',     # JAN-15-2024
    r'USE BY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Use by date
        r'BEST BEFORE[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Best before date
        r'EXPIRY DATE[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiry Date
        r'EXPIRY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiry
        r'EXP[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Exp
        r'VALID UNTIL[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Valid Until
        r'CONSUME BY[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Consume By
        r'EXPIRES ON[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expires On
        # DDMMMYYYY format
        r'(\d{1,2}[A-Za-z]{3}\d{4})',  # DDMMMYYYY format
        # Short year formats
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',  # Short year date format (DD/MM/YY)
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # General date format (DD/MM/YYYY)
        # Year-month-day formats
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',  # Year-month-day format (YYYY/MM/DD)
        # Month/Year formats
        r'(\d{1,2}[\/\-]\d{1,2})',  # MM/DD format
        r'(\d{1,2}[\/\-]\d{2})',  # MM/YY format
        # Month name formats
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})',  # Month name with day and year
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{2})',  # Month name with day and short year
        # Year with month name
        r'(\d{4}[A-Za-z]{3,}\d{1,2})',  # Year with month name
        r'(\d{1,2}[A-Za-z]{3,}\d{4})',  # Day with month name and full year
        # Additional formats
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',  # MM/DD/YY format
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # MM/DD/YYYY format
        r'(\d{1,2}[\/\-]\d{1,2})',  # MM/DD format
        r'(\d{1,2}[\/\-]\d{2})',  # MM/YY format
        # Best before phrases
        r'Best before (\d+) months',  # Best before in months
        r'Expiration Date[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiration Date
        r'Expires[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expires
        # Additional variations
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})',  # Month name with day and year
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{2})',  # Month name with day and short year
        # More variations
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # MM/DD/YYYY format
        r'(\d{1,2}[\/\-]\d{1,2})',  # MM/DD format
        r'(\d{1,2}[\/\-]\d{2})',  # MM/YY format
        # Additional expiry phrases
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiry in various formats
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',  # Expiry in short year formats
        r'(\d{1,2}[\/\-]\d{1,2})',  # Expiry in MM/DD format
        r'(\d{1,2}[\/\-]\d{2})',  # Expiry in MM/YY format
        # Additional phrases
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})',  # Month name with day and year
        r'(\d{1,2}\s*[A-Za-z]{3,}\s*\d{2})',  # Month name with day and short year
        r'(\d{4}[A-Za-z]{3,}\d{1,2})',  # Year with month name
        r'(\d{1,2}[A-Za-z]{3,}\d{4})',  # Day with month name and full year
        # Additional expiry phrases
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',  # Expiry in various formats
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',  # Expiry in short year formats
        r'(\d{1,2}[\/\-]\d{1,2})',  # Expiry in MM/DD format
        r'(\d{1,2}[\/\-]\d{2})',  # Expiry in MM/YY format
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)

    unique_dates = sorted(dates, key=lambda x: len(x), reverse=True)
    return [unique_dates[0]] if unique_dates else []  # Return the most likely date

# Helper function: Preprocess image for fruit freshness
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Helper function: Calculate days to expiry
def calculate_days_to_expiry(expiry_dates):
    results = []
    today = datetime.now()
    for date_str in expiry_dates:
        try:
            if '/' in date_str or '-' in date_str:
                if len(date_str.split('/')[-1]) == 2 or len(date_str.split('-')[-1]) == 2:
                    date_obj = datetime.strptime(date_str, '%d/%m/%y') if '/' in date_str else datetime.strptime(date_str, '%d-%m-%y')
                else:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y') if '/' in date_str else datetime.strptime(date_str, '%d-%m-%Y')
            else:
                date_obj = datetime.strptime(date_str, '%d %B %Y')

            delta = (date_obj - today).days
            if delta >= 0:
                results.append(f"{date_str}: {delta} days to expire")
            else:
                results.append(f"{date_str}: Expired")
        except ValueError:
            results.append(f"{date_str}: Invalid date format")
    return results

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Streamlit App
st.title("Flipkart Grid")

# Sidebar options
app_mode = st.sidebar.selectbox(
    "Choose the mode",
    ["Home", "Brand & Text Detection", "Fruit Freshness Detection", "Object Detection"],
    on_change=lambda: st.session_state.update({'uploaded_file': None})  # Reset uploaded file on mode change
)

if app_mode == "Home":
    st.markdown("""
    ## Welcome to the Detection App
    Use the sidebar to choose between:
    - *Brand & Text Detection*: Detect brands, extract text, and identify expiry dates from uploaded images.
    - *Fruit Freshness Detection*: Detect and classify the freshness of fruits from uploaded images.
    - *Object Detection*: Detect objects in uploaded images.
    """)

else:
    st.header(app_mode)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=app_mode)

    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file

    if st.session_state['uploaded_file'] is not None:
        file_bytes = np.asarray(bytearray(st.session_state['uploaded_file'].read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if app_mode == "Brand & Text Detection":
            # Brand detection
            results = brand_model.predict(source=image, stream=False)
            detected_image = results[0].plot()

            # OCR for text extraction
            _, img_buffer = cv2.imencode('.jpg', image)
            ocr_result = ocr.ocr(img_buffer.tobytes())
            if ocr_result and isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                extracted_text = ' '.join([line[1][0] for line in ocr_result[0]])
                expiry_dates = extract_expiry_dates(extracted_text)
                expiry_info = calculate_days_to_expiry(expiry_dates)
            else:
                extracted_text = "No text detected"
                expiry_dates = []
                expiry_info = []

            # Count objects
            object_counts = {}
            for box in results[0].boxes.data.cpu().numpy():
                label = results[0].names[int(box[5])]
                object_counts[label] = object_counts.get(label, 0) + 1

            # Display results
            st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Image")
            st.markdown(f"*Extracted Text:* {extracted_text}")
            st.markdown(f"*Expiry Dates:*")
            if expiry_info:
                for info in expiry_info:
                    st.markdown(f"- {info}")
            else:
                st.markdown("None")
            st.markdown("*Object Counts:*")
            for label, count in object_counts.items():
                st.markdown(f"- {label}: {count}")

        elif app_mode == "Fruit Freshness Detection":
            # Preprocess and predict
            img_array = preprocess_image(image)
            predictions = fruit_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            label = class_names[predicted_class]
            confidence = predictions[0][predicted_class] * 100

            # Display results
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")
            st.markdown(f"*Label:* {label}")
            st.markdown(f"*Confidence:* {confidence:.2f}%")

        elif app_mode == "Object Detection":
            # Object Detection
            results = object_model.predict(source=image, stream=False)
            detected_objects = []

            for result in results:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    class_id = int(box[5])
                    confidence = box[4]  # Assuming the confidence score is at index 4
                    detected_objects.append((result.names[class_id], confidence))

                    # Draw bounding box and label on the image
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = f"{result.names[class_id]} {confidence * 100:.2f}%"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the image back to RGB for display in Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption='Detected Objects', use_container_width=True)

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
