import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import gym_functions as gf

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

#################################################################
# Function to detect the current push-up state based on the angles
#################################################################
def get_pushup_state(elbow_angle):
    if elbow_angle > 160:
        return "Up Position"
    elif 100 <= elbow_angle <= 160:
        return "Lowering"
    elif elbow_angle < 100:
        return "Down Position"
    return "Unknown"

#########################################
# Function to count the reps over a video
#########################################
def calc_pushup_reps(pushup_state_frame, pushup_state_memory, pushup_state, counters):
    if pushup_state_frame == 0:
        pushup_state_memory.append(pushup_state)
        pushup_state_frame += 1
    else:
        if pushup_state != pushup_state_memory[pushup_state_frame - 1]:
            if pushup_state != 'Up Position':
                pushup_state_memory.append(pushup_state)
                pushup_state_frame += 1
            else:
                pushup_state_memory.append(pushup_state)
                # print(pushup_state_memory)
                if 'Down Position' in pushup_state_memory:
                    counters['correct_reps'] += 1
                else:
                    counters['wrong_reps'] += 1
                pushup_state_memory = []
                pushup_state_frame = 0
    return pushup_state_frame, pushup_state_memory, pushup_state, counters

#############################################
# Function to display push-up workout results
#############################################
def process_pushup_frame(image, results, pushup_state_frame, pushup_state_memory, counters):
    """Process each frame for push-up workout detection, count reps, and annotate the frame."""
    
    # Initialize push-up state
    pushup_state = "Unknown"
    elbow_angle_pos = 15
    angles = []

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Detect orientation using gym_functions' detect_orientation
        orientation = gf.detect_orientation(landmarks)
        cv2.putText(image, f'Orientation: {orientation}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)

        # If front orientation, show keypoints for both arms
        if orientation == "Front":
            keypoint_pairs = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
            ]
            keypoints = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
            ]

        # Else, show keypoints for only the side arm
        else:
            if orientation == "Left Side":
                keypoint_pairs = [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)
                ]
                keypoints = [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)]
            else:
                keypoint_pairs = [
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
                ]
                keypoints = [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)]

        # Draw connections between the appropriate shoulder, elbow, and wrist
        for pair in keypoint_pairs:
            point1 = (int(landmarks[pair[0].value].x * image.shape[1]), int(landmarks[pair[0].value].y * image.shape[0]))
            point2 = (int(landmarks[pair[1].value].x * image.shape[1]), int(landmarks[pair[1].value].y * image.shape[0]))
            cv2.line(image, point1, point2, (255, 255, 0), 2)

        # Draw bitrolls and calculate angles for the arms
        for shoulder, elbow, wrist in keypoints:
            shoulder_point = (int(landmarks[shoulder.value].x * image.shape[1]), int(landmarks[shoulder.value].y * image.shape[0]))
            elbow_point = (int(landmarks[elbow.value].x * image.shape[1]), int(landmarks[elbow.value].y * image.shape[0]))
            wrist_point = (int(landmarks[wrist.value].x * image.shape[1]), int(landmarks[wrist.value].y * image.shape[0]))

            gf.draw_bitroll(image, shoulder_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)
            gf.draw_bitroll(image, elbow_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)
            gf.draw_bitroll(image, wrist_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)

            angle = gf.calculate_angle(shoulder_point, elbow_point, wrist_point)
            angles.append(angle)
            cv2.putText(image, str(round(angle, 2)), (elbow_point[0] + elbow_angle_pos, elbow_point[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elbow_angle_pos *= (-5)

        # Detect push-up state
        pushup_state = get_pushup_state(angles[0])

        # Count the wrong and the right reps
        pushup_state_frame, pushup_state_memory, pushup_state, counters = calc_pushup_reps(pushup_state_frame, pushup_state_memory, pushup_state, counters)

        # Display push-up state and counters
        cv2.putText(image, f'Right Reps: {counters.correct_reps}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Wrong Reps: {counters.wrong_reps}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    return image, pushup_state_frame, pushup_state_memory, counters


#################################
# Initialize the Push-ups Counter
#################################
def pushups_counters():
    reps_dict = {'correct_reps': 0, 'wrong_reps': 0}
    return pd.Series(reps_dict)


################################################
# Functions to detect the errors in the workouts
################################################
def check_body_alignment(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Extract the relevant landmarks for alignment: head, shoulders, hips, ankles
        head = [lm[0].x, lm[0].y]
        left_shoulder = [lm[11].x, lm[11].y]
        left_hip = [lm[23].x, lm[23].y]
        left_ankle = [lm[27].x, lm[27].y]
        
        # Calculate the angle between head, hip, and ankle to detect body misalignment
        body_angle = gf.calculate_angle(head, left_hip, left_ankle)
        
        # If the body is not straight (e.g., angle deviates too much), alignment is off
        if body_angle < 170:  # Example threshold for poor alignment
            return False
        return True


def check_elbow_angle(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Extract the relevant landmarks: shoulder, elbow, and wrist
        left_shoulder = [lm[11].x, lm[11].y]
        left_elbow = [lm[13].x, lm[13].y]
        left_wrist = [lm[15].x, lm[15].y]
        
        # Calculate the elbow angle
        elbow_angle = gf.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # If the elbow is not bent properly (e.g., angle is too high), poor form is detected
        if elbow_angle > 160:  # Threshold for not bending enough
            return False
        return True


def check_hip_sagging(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Extract the relevant landmarks: shoulder, hip, and knee
        left_shoulder = [lm[11].x, lm[11].y]
        left_hip = [lm[23].x, lm[23].y]
        left_knee = [lm[25].x, lm[25].y]
        
        # Calculate the angle between shoulder, hip, and knee to detect sagging
        hip_angle = gf.calculate_angle(left_shoulder, left_hip, left_knee)
        
        # If the hips are sagging (angle is too low), poor form is detected
        if hip_angle < 160:  # Example threshold for sagging
            return False
        return True


def check_push_up_errors(landmarks, errors=None):
    # Initialize error dictionary if it's empty or not provided
    if len(errors.keys()) == 0:
        errors = {
            'body_alignment': 0,
            'elbow_bend': 0,
            'hip_sagging': 0
        }
    
    # Check for body alignment issues
    if not check_body_alignment(landmarks):
        errors['body_alignment'] += 1
    
    # Check for elbow bending issues
    if not check_elbow_angle(landmarks):
        errors['elbow_bend'] += 1
    
    # Check for hip sagging issues
    if not check_hip_sagging(landmarks):
        errors['hip_sagging'] += 1
    
    return errors
