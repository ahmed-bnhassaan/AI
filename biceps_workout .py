import cv2
import numpy as np
import mediapipe as mp

import gym_functions as gf
# from gym_functions import get_curl_state, calc_curl_reps

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

#################################################################
# Function to detect the current bicep curl state based on the angles
#################################################################
def get_curl_state(elbow_angle, side_view=True):
    if side_view:
        # Side view angle ranges
        if elbow_angle > 150:
            return "Extended"
        elif 75 <= elbow_angle <= 150:
            return "Curling"
        elif elbow_angle < 75:
            return "Fully Flexed"
        
    else:
        # Front view angle ranges (similar or adjusted if needed)
        if elbow_angle > 150:
            return "Extended"
        elif 60 <= elbow_angle <= 150:
            return "Curling"
        elif elbow_angle < 60:
            return "Fully Flexed"
       
    return "Unknown"


#########################################
# Function to count the reps over a video
#########################################
def calc_curl_reps(curl_state_frame, curl_state_memory, curl_state, counters, arm):
    # Check if the memory list for the specific arm is empty
    if len(curl_state_memory[arm]) == 0:
        # Append the current curl state to the memory list
        curl_state_memory[arm].append(curl_state)
        curl_state_frame += 1
    else:
        # Check if the current state is different from the last state
        if curl_state != curl_state_memory[arm][-1]:
            if curl_state != 'Extended':
                curl_state_memory[arm].append(curl_state)
                curl_state_frame += 1
            else:
                # If the arm is extended, check the memory and count the rep
                curl_state_memory[arm].append(curl_state)
                # print(curl_state_memory)  # For debugging
                if 'Fully Flexed' in curl_state_memory[arm]:
                    counters[f'correct_reps_{arm}'] += 1
                else:
                    counters[f'wrong_reps_{arm}'] += 1
                # Reset memory and frame for the next rep
                curl_state_memory[arm] = []
                curl_state_frame = 0
    return curl_state_frame, curl_state_memory, curl_state, counters




#####################################################
# Function to process bicep curls and display results
#####################################################
def process_curl_frame(image, results, curl_state_frame, curl_state_memory, counters):
    """Process each frame for bicep curls, count reps, and annotate the frame."""

    # Initialize curl states for both arms
    curl_state_left = "Unknown"
    curl_state_right = "Unknown"
    elbow_angle_pos = 15
    angles = {'left': [], 'right': []}

    # Draw landmarks and highlight shoulders, elbows, and wrists
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Detect orientation and determine curl states for each arm
        orientation = gf.detect_orientation(landmarks)
        is_side_view = "side" in orientation
        
        # Show both arms for front view
        show_left = "left" in orientation or "front" in orientation
        show_right = "right" in orientation or "front" in orientation

        # Define key points for both arms (shoulder, elbow, wrist)
        keypoints = {
            'left': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            'right': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
        }

        # Define line connections for left and right arms
        connections = {
            'left': [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                     (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)],
            'right': [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                      (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)]
        }

        # Draw keypoints and lines based on the detected orientation
        for side in ['left', 'right']:
            if (side == 'left' and show_left) or (side == 'right' and show_right):
                shoulder, elbow, wrist = keypoints[side]

                # Get coordinates of the landmarks
                shoulder_point = (int(landmarks[shoulder.value].x * image.shape[1]), int(landmarks[shoulder.value].y * image.shape[0]))
                elbow_point = (int(landmarks[elbow.value].x * image.shape[1]), int(landmarks[elbow.value].y * image.shape[0]))
                wrist_point = (int(landmarks[wrist.value].x * image.shape[1]), int(landmarks[wrist.value].y * image.shape[0]))

                # Draw bitrolls (circles) on shoulders, elbows, and wrists
                gf.draw_bitroll(image, shoulder_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)
                gf.draw_bitroll(image, elbow_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)
                gf.draw_bitroll(image, wrist_point, outer_radius=10, inner_radius=7, color=(0, 255, 255), thickness=2)

                # Draw lines between the shoulder, elbow, and wrist
                for pair in connections[side]:
                    point1 = (int(landmarks[pair[0].value].x * image.shape[1]), int(landmarks[pair[0].value].y * image.shape[0]))
                    point2 = (int(landmarks[pair[1].value].x * image.shape[1]), int(landmarks[pair[1].value].y * image.shape[0]))
                    cv2.line(image, point1, point2, (255, 255, 0), 2)

                # Calculate elbow angle and display it
                angle = gf.calculate_angle(shoulder_point, elbow_point, wrist_point)
                angles[side].append(angle)
                cv2.putText(image, str(round(angle, 2)), (elbow_point[0] + elbow_angle_pos, elbow_point[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elbow_angle_pos *= (-5)

        # Determine curl states for each arm
        if show_left:
            curl_state_left = get_curl_state(angles['left'][0], side_view=is_side_view)
            curl_state_frame, curl_state_memory, curl_state_left, counters = calc_curl_reps(curl_state_frame, curl_state_memory, curl_state_left, counters, arm='left')

        if show_right:
            curl_state_right = get_curl_state(angles['right'][0], side_view=is_side_view)
            curl_state_frame, curl_state_memory, curl_state_right, counters = calc_curl_reps(curl_state_frame, curl_state_memory, curl_state_right, counters, arm='right')

        # Check if both arms are active
        both_arms_active = (curl_state_left != "Extended" and curl_state_right != "Extended")
        both_arms_status = "Both Arms Active" if both_arms_active else "One Arm Active"

        # Display curl states, orientation, and counters for correct and wrong reps
        cv2.putText(image, f'Orientation: {orientation}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        if show_left:
            left_arm_wrong_pos = 250
            if "left" in orientation:
                left_arm_wrong_pos = 200
            cv2.putText(image, f'Left Arm Correct Reps: {counters["correct_reps_left"]}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Left Arm Wrong Reps: {counters["wrong_reps_left"]}', (50, left_arm_wrong_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        if show_right:
            right_arm_correct_pos = 200
            right_arm_wrong_pos = 300
            if "right" in orientation:
                right_arm_correct_pos = 150
                right_arm_wrong_pos = 200
            cv2.putText(image, f'Right Arm Correct Reps: {counters["correct_reps_right"]}', (50, right_arm_correct_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Arm Wrong Reps: {counters["wrong_reps_right"]}', (50, right_arm_wrong_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
        if both_arms_status== 'Both Arms Active':
            arms_status_pos = 350
            if "front" not in orientation:
                arms_status_pos = 250
            # Display arm activity status
            cv2.putText(image, both_arms_status, (50, arms_status_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 0), 2, cv2.LINE_AA)

    return image, curl_state_frame, curl_state_memory, counters


#####################################
# Initialize the Biceps Counter
#####################################
def biceps_counters():
    return {
        'correct_reps_left': 0,
        'wrong_reps_left': 0,
        'correct_reps_right': 0,
        'wrong_reps_right': 0
    }


################################################
# functions to detect the errors in the workouts
################################################
def check_swinging(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Extract shoulder and hip landmarks
        left_shoulder = [lm[11].x, lm[11].y]
        left_hip = [lm[23].x, lm[23].y]
        
        # Calculate angle between the shoulder and hip to detect torso movement
        torso_angle = gf.calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] + 0.1])  # using a straight line for hip to detect sway
        
        # If there's significant torso movement (e.g., torso angle deviates too much), swinging is detected
        if torso_angle < 160:  # Example threshold, adjust as needed
            return False
        return True


def check_elbow_movement(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Extract elbow and shoulder landmarks
        left_elbow_x = lm[13].x
        left_shoulder_x = lm[11].x
        
        # If the elbow moves too far from the shoulder's x-position, it's considered poor form
        if abs(left_elbow_x - left_shoulder_x) > 0.05:  # Adjust threshold for proper distance
            return False
        return True



def check_back_arch(landmarks):
    if landmarks.pose_landmarks:
        lm = landmarks.pose_landmarks.landmark
        
        # Extract shoulder, hip, and knee landmarks
        left_shoulder = [lm[11].x, lm[11].y]
        left_hip = [lm[23].x, lm[23].y]
        left_knee = [lm[25].x, lm[25].y]
        
        # Calculate the angle between shoulder, hip, and knee to detect back arching
        back_angle = gf.calculate_angle(left_shoulder, left_hip, left_knee)
        
        # If the angle is too wide (>160 degrees), it indicates the back is arching
        if back_angle < 160:  # Example threshold for arching
            return False
        return True



def check_bicep_curl_errors(landmarks, errors=None):
    # Initialize error dictionary if it's empty or not provided
    if len(errors.keys()) == 0:
        errors = {
            'swinging': 0,
            'elbow_movement': 0,
            'back_arch': 0
        }
    
    # Check for swinging issues
    if not check_swinging(landmarks):
        errors['swinging'] += 1
    
    # Check for elbow movement issues
    if not check_elbow_movement(landmarks):
        errors['elbow_movement'] += 1
    
    # Check for back arching issues
    if not check_back_arch(landmarks):
        errors['back_arch'] += 1
    
    return errors


