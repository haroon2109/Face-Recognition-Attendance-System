Facial Recognition Attendance System (FRAS)
A Python-based facial recognition system for automated attendance tracking using a webcam. The project is built with OpenCV for computer vision, scikit-learn for machine learning, and other standard Python libraries.

🚀 Features
Face Data Collection: Uses a webcam to capture facial images for new users.

Real-time Recognition: Identifies registered users in real-time using a K-Nearest Neighbors (KNN) classifier.

Automated Attendance: Records attendance with a timestamp upon a key press.

Data Export: Saves attendance logs to a date-stamped CSV file.

Text-to-Speech (Windows Only): Announces when attendance is successfully taken.

Progress Visualization: Displays a real-time progress bar during the face data collection process.

🛠️ Prerequisites
Before you run the project, ensure you have Python installed on your system. You'll also need the following libraries:

opencv-python

numpy

scikit-learn

win32com (for Windows users only, for text-to-speech)

You can install all the required libraries by running the following command in your terminal:

pip install opencv-python numpy scikit-learn win32com

📂 Project Structure
FRAS_Project/
├── datatrain.py              # Script to collect and save face data
├── AttendanceSystem.py       # Script to perform face recognition and log attendance
├── data/
│   └── haarcascade_frontalface_default.xml  # Pre-trained face detection model
├── Attendance/               # Directory to store attendance CSV files (created automatically)
└── README.md                 # Project description and instructions

⚙️ How to Use
Step 1: Collect Face Data
First, you need to collect facial data for the people you want the system to recognize.

Run the datatrain.py script from your terminal:

python datatrain.py

The script will prompt you to "Enter Your Name". Type a name and press Enter.

A webcam window will open and begin capturing 100 face samples. A progress bar will show the collection status.

Once 100 samples are collected, the window will close automatically.

Step 2: Run the Attendance System
After collecting the face data, you can run the main attendance system.

Run the AttendanceSystem.py script:

python AttendanceSystem.py

A webcam window will open. The system will detect and try to recognize faces, displaying the recognized name.

To mark a person's attendance, ensure their face is detected and press the 'o' key. The attendance will be logged in a CSV file.

To exit the program, press the 'q' key.

📝 Attendance Logs
Attendance records are saved in a CSV file inside the Attendance/ folder. The file name is based on the current date (e.g., Attendance_23-08-2025.csv). Each row in the file contains the name of the person and the time their attendance was taken.
