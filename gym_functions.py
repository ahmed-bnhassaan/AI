import numpy as np
import pandas as pd
import cv2
import math
import mediapipe as mp
import statistics
import joblib
import time

import squat_workout as sw
import biceps_workout as bw
import pushups_workout as pw

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


###################################################
# Function to calculate the angles between 3 points
###################################################
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (hip)
    b = np.array(b)  # Middle point (knee)
    c = np.array(c)  # Last point (ankle)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle


################################################
# Function to find the distance between 2 points
################################################
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

################################################
# Function to find the side of the trainee
################################################
def detect_orientation(landmarks, thresh=0.6):
    # Extract landmarks
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    
    # Calculate distances
    shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
    
    # Determine visibility and calculate distance for the most visible side
    if left_shoulder.visibility > right_shoulder.visibility:
        shoulder_to_elbow_distance = calculate_distance(left_shoulder, left_elbow)
        dominant_side = "left"
    else:
        shoulder_to_elbow_distance = calculate_distance(right_shoulder, right_elbow)
        dominant_side = "right"
    
    # Calculate the ratio
    if shoulder_to_elbow_distance != 0:  # Avoid division by zero
        ratio = shoulder_distance / shoulder_to_elbow_distance
    else:
        return "unknown"  # Handle edge case where shoulder distance is zero

    # Determine orientation based on the ratio
    if ratio >= thresh:
        return "front"
    else:
        return f"{dominant_side} side"


###########################################
# Function to draw a bitroll (donut shape)
###########################################
def draw_bitroll(image, center, outer_radius, inner_radius, color, thickness=1):
    # Draw outer circle (the bitroll)
    cv2.circle(image, center, outer_radius, color, thickness=thickness)
    # Draw a filled smaller circle inside (the hole of the bitroll)
    cv2.circle(image, center, inner_radius, color, thickness=-1)



######################################################################################
# Function to Extract points coordinates to the ML model to detect the current workout
####################################################################################### 
def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(99)  # 33 landmarks * 3 coordinates (x, y, z)
    return keypoints



########################################
# Function to Detect the current workout
########################################
# Function to perform workout detection and display results
def workout_detection( clf,scaler,frame, frames_counter, workout_for_Second, current_workout):
    # Convert the frame to RGB (MediaPipe works with RGB images)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(image_rgb)

    # Extract keypoints from the frame
    keypoints = extract_keypoints(results)

    # Reshape and scale the keypoints
    keypoints_reshaped = np.expand_dims(keypoints, axis=0)
    keypoints_scaled = scaler.transform(keypoints_reshaped)

    # Make a prediction using the trained model
    prediction = clf.predict(keypoints_scaled)
    predicted_class = prediction[0]
    
    # Append the prediction to the workout list
    workout_for_Second.append(predicted_class)
    frames_counter += 1
    # Display the predicted class on the frame
    cv2.putText(frame, f'Workout: {statistics.mode(current_workout)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2, cv2.LINE_AA)
    
    # Every 10 frames, update the current workout with the mode of predictions
    if len(current_workout) < 5:
        if frames_counter % 10 == 0:
            current_workout.append(statistics.mode(workout_for_Second))
            workout_for_Second = []
            frames_counter = 0
    elif len(current_workout) == 5:
        if statistics.mode(current_workout) == 'Neutral':
            current_workout = ['Neutral']
        else:
            if frames_counter % 10 == 0:
                current_workout = current_workout[1:]
                current_workout.append(statistics.mode(workout_for_Second))
                workout_for_Second = []
                frames_counter = 0
    # # Draw landmarks on the frame
    # if results.pose_landmarks:
    #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame, frames_counter, workout_for_Second, current_workout


def release_assistant(vid_path, st_frame_placeholder):
    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize memory lists to store workout states over time
    squat_state_memory = []
    squat_state_frame = 0
    curl_state_memory = {'left': [], 'right': []}
    curl_state_frame = 0
    pushup_state_memory = []
    pushup_state_frame = 0

    errors = {}

    # Initialize the Counters for the reps (for squats, curls, and push-ups)
    counters = ''

    # Load the trained model and scaler
    clf = joblib.load('workout_classifier_model.joblib')
    scaler = joblib.load('scaler.joblib')

    # Open the video file
    cap = cv2.VideoCapture(vid_path)

    # Initialize counters and storage
    frames_counter = 0
    workout_for_Second = []
    current_workout = ['Neutral']

    # Loop through video frames
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Resize the image if needed
        if type(vid_path) == int and vid_path > 0:
            image = cv2.resize(image, (image.shape[0] * 2, image.shape[1]))

        # Call the workout detection function
        image, frames_counter, workout_for_Second, current_workout = workout_detection(
            clf, scaler, image, frames_counter, workout_for_Second, current_workout)

        # Convert the frame to RGB for pose detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Handle Squats
        if statistics.mode(current_workout) == 'Squats':
            if type(counters) == str:
                counters = sw.squats_counters()
            image, squat_state_frame, squat_state_memory, counters = sw.process_squat_frame(
                image, results, squat_state_frame, squat_state_memory, counters)
            
            # Find the workout errors
            errors = sw.check_squat_errors(results, errors)

        # Handle Bicep Curls
        elif statistics.mode(current_workout) == 'Bicep-curls':
            if type(counters) == str:
                counters = bw.biceps_counters()
            image, curl_state_frame, curl_state_memory, counters = bw.process_curl_frame(
                image, results, curl_state_frame, curl_state_memory, counters)
                        
            # Find the workout errors
            errors = bw.check_bicep_curl_errors(results, errors)

        # Handle Push-Ups
        elif statistics.mode(current_workout) == 'Push-ups':
            if type(counters) == str:
                counters = pw.pushups_counters()
            image, pushup_state_frame, pushup_state_memory, counters = pw.process_pushup_frame(
                image, results, pushup_state_frame, pushup_state_memory, counters)
            
            # Find the workout errors
            errors = pw.check_push_up_errors(results, errors)

        # Resize frame and display it on Streamlit
        image_resized = cv2.resize(image, (640, 480))
        st_frame_placeholder.image(image_resized, channels="BGR")

        # Optional: Add delay for visualization to simulate real-time (can be adjusted)
        # time.sleep(0.003)  # 30ms delay (for a smooth frame rate)

    cap.release()

    # Return final data
    workout_class = statistics.mode(current_workout)
    return counters, workout_class, errors



