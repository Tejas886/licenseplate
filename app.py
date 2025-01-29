import cv2
import pytesseract
import numpy as np
import mysql.connector
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)  # FIXED: Changed _name_ to __name__
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Database connection settings (Update these for your XAMPP MySQL)
db_config = {
    "host": "localhost",
    "user": "root",         # Default user for XAMPP
    "password": "",         # Leave empty if no password
    "database": "license_plate_db"
}

# Ensure Tesseract is correctly installed (update path if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows users update this

def connect_db():
    """ Connect to MySQL database """
    return mysql.connector.connect(**db_config)

def preprocess_image(image_path):
    """ Preprocess the image to improve OCR accuracy """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Detect edges
    edged = cv2.Canny(threshold, 30, 200)
    
    # Find contours and extract possible license plate regions
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    possible_plates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        if 2 < aspect_ratio < 6:  # Typical aspect ratio for license plates
            possible_plates.append((x, y, w, h))
    
    if possible_plates:
        possible_plates = sorted(possible_plates, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = possible_plates[0]
        cropped_plate = gray[y:y+h, x:x+w]
        return cropped_plate
    else:
        return gray

def extract_license_plate(image_path):
    """ Extract license plate text using OCR """
    processed_image = preprocess_image(image_path)
    
    # Use Tesseract OCR with better configurations
    plate_text = pytesseract.image_to_string(processed_image, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    
    return plate_text.strip()

def save_to_database(image_name, plate_number):
    """ Save detected license plate to MySQL database """
    conn = connect_db()
    cursor = conn.cursor()
    
    query = "INSERT INTO plates (image_name, plate_number) VALUES (%s, %s)"
    cursor.execute(query, (image_name, plate_number))
    
    conn.commit()
    cursor.close()
    conn.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    plate_number = extract_license_plate(file_path)
    
    if not plate_number:
        return jsonify({"error": "Could not detect license plate"}), 400
    
    # Save to MySQL
    save_to_database(file.filename, plate_number)
    
    return jsonify({"plate_number": plate_number})

if __name__ == "__main__":  # FIXED: Changed _name_ to __name__
    app.run(debug=True)
