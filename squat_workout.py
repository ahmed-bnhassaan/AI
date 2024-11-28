import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

import gym_functions as gf
# from gym_functions import get_squat_state, calc_squat_reps

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

#################################################################
# Function to detect the current squats state based on the angles
#################################################################
def get_squat_state(knee_angle, side_view=True):
    if side_view:
        # Side view angle ranges
        if knee_angle > 160:
            return "Standing"
        elif 90 <= knee_angle <= 160:
            return "Lowering"
        elif knee_angle < 90:
            return "Bottom Position"
        
    else:
        # Front view angle ranges (similar or adjusted if needed)
        if knee_angle > 160:
            return "Standing"
        elif 150 <= knee_angle <= 160:
            return "Lowering"
        elif knee_angle < 150:
            return "Bottom Position"
       
    return "Unknown"


#########################################
# Function to count the reps over a video
#########################################
def calc_squat_reps(squat_state_frame,squat_state_memory,squat_state,counters):
    # Append every unique state in the reps to the memory
    if squat_state_frame == 0:
        # Append the current squat state to the memory list
        squat_state_memory.append(squat_state)
        squat_state_frame += 1
    else:
        if squat_state != squat_state_memory[squat_state_frame - 1]:
            if squat_state != 'Standing':
                squat_state_memory.append(squat_state)
                squat_state_frame += 1
            else:
                squat_state_memory.append(squat_state)
                # print(squat_state_memory)
                if 'Bottom Position' in squat_state_memory:
                    counters['correct_reps'] += 1
                else:
                    counters['wrong_reps'] += 1
                squat_state_memory = []
                squat_state_frame = 0
    return squat_state_frame,squat_state_memory,squat_state,counters


#############################################
# Function to display squats workout results
#############################################
def process_squat_frame(image, results, squat_state_frame, squat_state_memory, counters):
    """Process each frame for squat workout detection, count reps, and annotate the frame."""
    
    # Initialize squat state
    squat_state = "Unknown"
    knee_angle_pos = 15
    angles = []
    
    # Draw landmarks and highlight hips, knees, and ankles
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Define the connections between hips, knees, and ankles
        keypoint_pairs = [
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]

        # Define key points for both legs: hips, knees, and ankles
        keypoints = [
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        # Draw connections between the hips, knees, and ankles
        for pair in keypoint_pairs:
            point1 = (int(landmarks[pair[0].value].x * image.shape[1]), int(landmarks[pair[0].value].y * image.shape[0]))
            point2 = (int(landmarks[pair[1].value].x * image.shape[1]), int(landmarks[pair[1].value].y * image.shape[0]))
            cv2.line(image, point1, point2, (255, 255, 0), 2)

        # Draw bitrolls and calculate angles
        for hip, knee, ankle in keypoints:
            # Get coordinates of the landmarks
            hip_point = (int(landmarks[hip.value].x * image.shape[1]), int(landmarks[hip.value].y * image.shape[0]))
            knee_point = (int(landmarks[knee.value].x * image.shape[1]), int(landmarks[knee.value].y * image.shape[0]))
            ankle_point = (int(landmarks[ankle.value].x * image.shape[1]), int(landmarks[ankle.value].y * image.shape[0]))

            # Draw bitrolls on hips, knees, and ankles
            gf.draw_bitroll(image, hip_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)
            gf.draw_bitroll(image, knee_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)
            gf.draw_bitroll(image, ankle_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)

            # Calculate knee angle
            angle = gf.calculate_angle(hip_point, knee_point, ankle_point)
            angles.append(angle)
            cv2.putText(image, str(round(angle, 2)), (knee_point[0] + knee_angle_pos, knee_point[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            knee_angle_pos *= (-5)

        # Detect orientation and determine squat state
        orientation = gf.detect_orientation(landmarks)
        is_side_view = "side" in orientation
        squat_state = get_squat_state(angles[0], side_view=is_side_view)

        # Count the wrong and the right reps
        squat_state_frame, squat_state_memory, squat_state, counters = calc_squat_reps(squat_state_frame, squat_state_memory, squat_state, counters)

        # Display squat state, orientation, and counters
        cv2.putText(image, f'Orientation: {orientation}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Reps: {counters.correct_reps}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Wrong Reps: {counters.wrong_reps}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    return image, squat_state_frame, squat_state_memory, counters



#################################
# Initialize the Squats Counter
#################################

def squats_counters():
    reps_dict = {'correct_reps' : 0 , 'wrong_reps' : 0 }
    return pd.Series(reps_dict)



###########################################
# Functions TO detect errors in the workout
###########################################
# Check squat depth based on knee, hip, and ankle angles
def check_squat_depth(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Left side
        left_hip = [lm[24].x, lm[24].y]
        left_knee = [lm[26].x, lm[26].y]
        left_ankle = [lm[28].x, lm[28].y]
        
        # Calculate the left hip angle
        left_hip_angle = gf.calculate_angle(left_hip, left_knee, left_ankle)
        
        # Squat depth considered low if hip angle is above 100 degrees
        if left_hip_angle > 100:
            return False
        return True

# Check back posture using the shoulder, hip, and knee angle
def check_back_posture(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Left side
        left_shoulder = [lm[12].x, lm[12].y]
        left_hip = [lm[24].x, lm[24].y]
        left_knee = [lm[26].x, lm[26].y]
        
        # Calculate the left shoulder-hip-knee angle
        left_shoulder_hip_angle = gf.calculate_angle(left_shoulder, left_hip, left_knee)
        
        # Check if the torso is upright, ideal if angle > 160 degrees
        if left_shoulder_hip_angle > 160:
            return True
        return False

# Check knee alignment to ensure it doesn't extend too far past the toes
def check_knee_alignment(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Left side
        left_knee_x = lm[26].x
        left_ankle_x = lm[28].x
        
        # If the knee is more than 20px in front of the toes, there's an alignment issue
        if abs(left_knee_x - left_ankle_x) > 0.02:  # Adjusted for normalized coordinates
            return False
        return True

# Function to check squat errors and update error dictionary
def check_squat_errors(landmarks, errors=None):
    # Initialize error dictionary if it's empty or not provided
    if len(errors.keys()) == 0:
        errors = {
            'depth_issue': 0,
            'back_posture_issue': 0,
            'knee_alignment_issue': 0
        }
    
    # Check for depth issues
    if not check_squat_depth(landmarks):
        errors['depth_issue'] += 1
    
    # Check back posture issues
    if not check_back_posture(landmarks):
        errors['back_posture_issue'] += 1
    
    # Check knee alignment issues
    if not check_knee_alignment(landmarks):
        errors['knee_alignment_issue'] += 1
    
    return errors

