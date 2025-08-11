import pickle
import numpy as np
import cv2
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

# Function to speak text using text-to-speech
def speak(text):
    try:
        speaker = Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
    except Exception as e:
        print("Error in text-to-speech:", e)

# Check if the Attendance directory exists; if not, create it
attendance_dir = "Attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Load the face detection model
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load the face data and labels from pickle files
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Verify that the FACES data and LABELS are aligned
if len(LABELS) != FACES.shape[0]:
    print("Warning: Mismatch between number of faces and labels!")
    min_samples = min(len(LABELS), FACES.shape[0])
    LABELS = LABELS[:min_samples]
    FACES = FACES[:min_samples]

# Initialize the KNN classifier and train it
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Define column names for the attendance CSV
COL_NAMES = ['NAME', 'TIME']

# Main loop for face detection and attendance
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    attendance = None  # Reset attendance for each frame
    date = datetime.now().strftime("%d-%m-%Y")  # Current date
    timestamp = datetime.now().strftime("%H:%M:%S")  # Current time

    # Process detected faces
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

        # Store attendance details
        attendance = [str(output[0]), str(timestamp)]

    # Display instruction on the frame
    cv2.putText(frame, "Press 'o' to take attendance, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # Show the frame
    cv2.imshow("Attendance System", frame)

    k = cv2.waitKey(1) & 0xFF  # Use & 0xFF for cross-platform compatibility

    if k == ord('o') and attendance is not None:  # If 'o' is pressed and attendance data is available
        print(f"Attendance taken for {attendance[0]} at {attendance[1]}")
        speak("Attendance taken")

        # Write attendance to CSV
        try:
            file_path = f"{attendance_dir}/Attendance_{date}.csv"
            file_exists = os.path.isfile(file_path)
            with open(file_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)  # Write header if file doesn't exist
                writer.writerow(attendance)
            print(f"Attendance for {attendance[0]} saved in {file_path}")
        except PermissionError:
            print(f"Permission denied: Unable to write to {file_path}. Ensure the file is not open elsewhere.")
        time.sleep(1)  # Pause briefly after attendance is taken

    elif k == ord('q'):  # Exit on pressing 'q'
        break

# Release resources
video.release()
cv2.destroyAllWindows()