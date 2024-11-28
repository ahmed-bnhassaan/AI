import cv2
import pandas as pd
import numpy as np

def provide_feedback(workout, errors, counters):
    feedback = []

    # Add workout summary title
    feedback.append(f"\n*** Workout Summary ***\n")

    # Initialize counts
    total_reps = np.sum(counters)  # Assuming counters is a dictionary

    if workout == 'Squats':
        correct_reps = counters.get('correct_reps', 0)
        incorrect_reps = counters.get('wrong_reps', 0)

        # Calculate overall correctness
        overall_correctness = (correct_reps / total_reps) * 100 if total_reps > 0 else 0
        
        # Add workout summary to feedback
        feedback.append(f"Total squats performed: {total_reps}\n")
        feedback.append(f"Correct squats: {correct_reps}\n")
        feedback.append(f"Incorrect squats: {incorrect_reps}\n")
        feedback.append(f"Overall correctness: {overall_correctness:.2f}%\n")
        if overall_correctness == 100:
            feedback.append("\nExcellent work! All repetitions were performed correctly with perfect form.\n")

        # Add areas to improve based on errors
        if np.sum(list(errors.values()))>0:
            feedback.append("\n**Areas to improve**\n")
            if errors['depth_issue']>0:
                feedback.append(f"- Squat depth needs improvement.\n")
            if errors['back_posture_issue'] >0:
                feedback.append(f"- Back posture needs to be straighter.\n")
            if errors['knee_alignment_issue'] >0:
                feedback.append(f"- Your knees are moving too far beyond your toes during the exercise.\n\n")

            # Suggestions for improvement
            feedback.append("**Suggestions for improvement**\n")
            if errors['depth_issue']>0:
                feedback.append("- Aim to bend your knees to about 90 degrees for proper squat depth.\n")
            if errors['back_posture_issue'] > 0:
                feedback.append("- Keep your back straight by engaging your core muscles throughout the movement.\n")
            if errors['knee_alignment_issue'] > 0:
                feedback.append("- Focus on shifting your hips back and keeping your knees in line with your toes to reduce strain.\n")


    elif workout == 'Bicep-curls':
        counters = pd.Series(counters)
        correct_reps_left = counters.correct_reps_left
        correct_reps_right = counters.correct_reps_right
        incorrect_reps_left = counters.wrong_reps_left
        incorrect_reps_right = counters.wrong_reps_right
        total_reps = 0
        if correct_reps_left==0 or correct_reps_right == 0:
            total_reps = np.sum(counters)
        else:
            total_reps = int(np.sum(counters)/2)
        correct_reps = correct_reps_left + correct_reps_right
        incorrect_reps = incorrect_reps_left + incorrect_reps_right
        if correct_reps > total_reps:
            correct_reps = int(correct_reps/2)
            incorrect_reps = int(incorrect_reps/2)

        overall_correctness = (correct_reps / total_reps) * 100 if total_reps > 0 else 0

        # Add workout summary to feedback
        feedback.append(f"Total Bicep Curls performed: {total_reps}\n")
        feedback.append(f"Correct Bicep Curls: {correct_reps}\n")
        feedback.append(f"Incorrect Bicep Curls: {incorrect_reps}\n")
        feedback.append(f"Overall correctness: {overall_correctness:.2f}%\n")
        if overall_correctness == 100:
            feedback.append("\nExcellent work! All repetitions were performed correctly with perfect form.\n")

        if np.sum(list(errors.values()))>0:
            feedback.append("\n**Areas to improve**\n")
            if errors['elbow_movement'] > 0:
                feedback.append(f"- Elbow movement needs stability.\n")
            if errors['back_arch'] > 0:
                feedback.append(f"- Back posture needs to be straighter.\n")
            if errors['swinging'] > 0:
                feedback.append(f"- our body is swinging during the curls, reducing the effectiveness of the workout.\n\n")

            feedback.append("\n**Suggestions for improvement**\n")
            if errors['elbow_movement'] > 0:
                feedback.append("- Keep your elbows stable; avoid moving them during the curl.\n")
            if errors['back_arch'] > 0:
                feedback.append("- Keep your back straight by engaging your core muscles throughout the movement.\n")
            if errors['swinging'] > 0:
                feedback.append("- Keep your torso stable and engage your core while lifting lighter weights to focus on controlled, strict movement.\n")

    elif workout == 'Push-ups':
        correct_reps = counters.correct_reps
        incorrect_reps = counters.wrong_reps
        overall_correctness = (correct_reps / total_reps) * 100 if total_reps > 0 else 0

        # Add workout summary to feedback
        feedback.append(f"Total Push-ups performed: {total_reps}\n")
        feedback.append(f"Correct Push-ups: {correct_reps}\n")
        feedback.append(f"Incorrect Push-ups: {incorrect_reps}\n")
        feedback.append(f"Overall correctness: {overall_correctness:.2f}%\n")
        if overall_correctness == 100:
            feedback.append("\nExcellent work! All repetitions were performed correctly with perfect form.\n")

        feedback.append("\n**Areas to improve**\n")
        if errors['hip_sagging'] > 0:
            feedback.append(f"- Avoid hip sagging.\n")
        if errors['body_alignment'] > 0:
            feedback.append(f"- Your body is not aligned correctly, leading to a drop in form and increased stress on your lower back.\n")
        if errors['elbow_bend'] > 0:
            feedback.append(f"- Your elbows are not bending properly during the movement, causing improper form and reduced muscle engagement.\n")


        feedback.append("\n**Suggestions for improvement**\n")
        if errors['hip_sagging'] > 0:
            feedback.append("- Keep your core engaged to avoid letting your hips sag.\n")
        if errors['body_alignment'] > 0:
            feedback.append("- Focus on maintaining a straight line from your shoulders to your heels by engaging your core and glutes throughout the movement.\n")
        if errors['elbow_bend'] > 0:
            feedback.append("- Focus on lowering your body until your elbows form a 90-degree angle and ensure your arms track close to your body for optimal form and efficiency.\n")


    feedback.append("\nKeep up the great work, and continue focusing on maintaining this high level of performance!")

    # # Add general tips or motivational statements
    # feedback.append("\nKeep up the good work and continue to focus on your form!")

    return feedback


###########################################
# Function to display feedback on the frame
###########################################
def display_feedback_on_frame(frame, feedback, position=(50, 100), font_scale=0.7, color=(0, 255, 0), thickness=2):
    """Display feedback as text overlay on the frame."""
    y_offset = position[1]
    for i, text in enumerate(feedback):
        wrapped_lines = [text[j:j+60] for j in range(0, len(text), 60)]  # Wrap long lines
        for line in wrapped_lines:
            cv2.putText(frame, line, (position[0], y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y_offset += 30  # Increment y-offset for each line

    return frame
