import cv2 as cv
import math
import numpy as np
from ultralytics import YOLO
import cvzone
from cvzone.ColorModule import ColorFinder

# Input video path
input_path = "/Users/tayosmacbook/Desktop/Basketball Predictor/vid1.MOV"

# Load YOLO model for hoop detection
hoop_model = YOLO("/Users/tayosmacbook/Desktop/Basketball Predictor/my_model 2/my_model.pt")

# Initialize Color Finder for ball tracking
myColorFinder = ColorFinder(False)
hsvVals = {"hmin": 6, "smin": 105, "vmin": 0, "hmax": 13, "smax": 255, "vmax": 255}

# Load the video
video = cv.VideoCapture(input_path)

# Score tracking variables
curr_frame_count = 0
shot_attempts = 0
shot_makes = 0
last_made_shot_frame = 0
last_missed_shot_frame= 0 
shot_in_progress_frame = 0
posList = []
ball_was_above_hoop = False
shot_in_progress = False  # Track when a shot is in progress

# Default hoop position (updated dynamically)
hoop_x1, hoop_y1, hoop_x2, hoop_y2 = 500, 550, 600, 650  

while True:
    ret, frame = video.read()
    curr_frame_count += 1
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape  

    # Detecting Hoop using YOLO model
    hoop_results = hoop_model(frame)
    for result in hoop_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            if confidence > 0.6:
                hoop_x1, hoop_y1, hoop_x2, hoop_y2 = x1, y1, x2, y2

    # Draw detected hoop on screen
    cv.rectangle(frame, (hoop_x1, hoop_y1), (hoop_x2, hoop_y2), (255, 0, 0), 2)

    # Use color detection to track basketball
    imgColor, mask = myColorFinder.update(frame, hsvVals)
    imgContours, contours = cvzone.findContours(frame, mask, minArea=210)

    if contours:
        pos = contours[0]["center"]

        # Only track if the ball is in the top 40% of the screen
        if pos[1] < int(frame_height * 0.40):  
            posList.append(pos)

        # Detect start of a shot attempt (ball goes above hoop)
        if pos[1] < hoop_y1 and not shot_in_progress:
            shot_in_progress = True
            shot_attempts += 1 
            posList.clear()  # Clear screen at the start of the shot attempt

        # Detect Shot Made (Ball enters hoop)
        if hoop_x1 < pos[0] < hoop_x2 and hoop_y1 < pos[1] < hoop_y2:
            if (curr_frame_count - last_made_shot_frame > 60) and (curr_frame_count - last_missed_shot_frame > 60):
                shot_makes += 1
                last_made_shot_frame = curr_frame_count
                ball_was_above_hoop = False  # Reset after made shot
                shot_in_progress = False  # Reset shot state

        # Detect Shot Missed (Ball falls below screen after being above hoop)
        if pos[1] > hoop_y1 and shot_in_progress: 
            if (curr_frame_count - last_made_shot_frame > 1000) and (curr_frame_count - last_missed_shot_frame > 1300): 
                last_missed_shot_frame = curr_frame_count
                ball_was_above_hoop = False  # Reset after made shot
                shot_in_progress = False  # Reset shot state

    # Perform 25th-degree polynomial regression
    prediction_text = "Prediction: N/A"
    if len(posList) > 25:
        x_vals = np.array([p[0] for p in posList])
        y_vals = np.array([p[1] for p in posList])
        coefficients = np.polyfit(x_vals, y_vals, 25)  # 25th-degree polynomial

        # Unpack coefficients (assuming a 25th-degree polynomial)
        coeff_dict = {chr(65 + i): coefficients[i] for i in range(26)}
        
        # Predict the ball's y-position at the hoop's x-position
        predicted_y = sum(coeff_dict[chr(65 + i)] * (hoop_x1 ** (25 - i)) for i in range(26))
        predicted_x = sum(coeff_dict[chr(65 + i)] * (hoop_y1 ** (25 - i)) for i in range(26))
        prediction_text = "Prediction: Make" if hoop_y1 <= predicted_y <= hoop_y2 or hoop_x1 <= predicted_x <= hoop_x2 else "Prediction: Miss"

    # Draw trajectory
    for i, pos in enumerate(posList):
        cv.circle(frame, pos, 5, (0, 255, 0), cv.FILLED)
        if i > 0:
            cv.line(frame, pos, posList[i - 1], (0, 255, 0), 2)

    # Scoreboard Display
    cv.rectangle(frame, (50, 50), (350, 250), (0, 0, 0), -1)
    cv.putText(frame, f"Shots Made: {shot_makes}", (70, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(frame, f"Shot Attempts: {shot_attempts}", (70, 130), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display shooting percentage
    if shot_attempts > 0:
        shooting_percentage = math.trunc((shot_makes / shot_attempts) * 100)
        cv.putText(frame, f"Shooting %: {shooting_percentage}%", (70, 170), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display shot prediction
    cv.putText(frame, prediction_text, (70, 210), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    cv.imshow("Basketball Tracker", frame)

    
    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
cv.destroyAllWindows()
