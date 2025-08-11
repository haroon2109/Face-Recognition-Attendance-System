import cv2
import pickle
import numpy as np
import os

# Set up video capture and face detection
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load the face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Error: Could not load face detection model.")
    exit()

faces_data = []
i = 0
name = input("Enter Your Name: ")

# Create named window and set it to pop up in the foreground with medium size
cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Capture", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("Face Capture", 640, 480)

# Function to display progress bar on the frame
def draw_progress_bar(frame, progress, max_progress):
    bar_width = 300
    filled_width = int(bar_width * (progress / max_progress))
    bar_height = 30
    start_x = (frame.shape[1] - bar_width) // 2
    start_y = frame.shape[0] - 50

    # Draw the background of the bar
    cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (50, 50, 50), 2)

    # Draw the filled part of the bar
    cv2.rectangle(frame, (start_x, start_y), (start_x + filled_width, start_y + bar_height), (50, 205, 50), -1)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))  # Ensure resizing to a consistent shape
        # Save every 10th frame to the dataset
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1

        # Draw a rectangle around the detected face with enhanced styling
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display "Gathering face data for {name}" on the first line
    cv2.putText(frame, f"Gathering face data for {name}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2,
                cv2.LINE_AA)
    # Display the progress on the next line
    collected_data_text = f"Progress: {len(faces_data)}/100"
    cv2.putText(frame, collected_data_text, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    # Draw progress bar for face data collection
    draw_progress_bar(frame, len(faces_data), 100)

    # Show the frame with face detection and UI improvements
    cv2.imshow("Face Capture", frame)

    # Check for exit conditions
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Reshape face data to be consistent
faces_data = np.asarray(faces_data).reshape(100, -1)
data_path = 'data'

# Save names
names_file = os.path.join(data_path, 'names.pkl')
if not os.path.exists(names_file):
    names = [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save face data
faces_data_file = os.path.join(data_path, 'faces_data.pkl')
if not os.path.exists(faces_data_file):
    with open(faces_data_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_data_file, 'rb') as f:
        faces = pickle.load(f)

    # Check the shapes of the arrays before concatenation
    if faces.shape[1] != faces_data.shape[1]:
        print(f"Shape mismatch: faces has shape {faces.shape}, faces_data has shape {faces_data.shape}")
        print("Reshaping or reprocessing of existing data is required to match dimensions.")
    else:
        # Append only when shapes match
        faces = np.append(faces, faces_data, axis=0)

    # Save the updated data
    with open(faces_data_file, 'wb') as f:
        pickle.dump(faces, f)

# Print confirmation of samples collected
print(f"Face samples collected for {name}.")