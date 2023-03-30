import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define a function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Define main function to capture video and analyze arm angles
def main():
    # Capture video from default camera
    cap = cv2.VideoCapture(0)

    # Use MediaPipe Pose model for pose estimation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Read video frames
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Detect pose landmarks
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract joint coordinates from pose landmarks
            if results.pose_landmarks:
                joints = []
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    joints.append((idx, x, y))
                
                # Calculate angles for right and left arms
                right_shoulder, right_elbow, right_wrist = joints[12], joints[14], joints[16]
                left_shoulder, left_elbow, left_wrist = joints[11], joints[13], joints[15]

                angle_r = calculate_angle(right_shoulder[1:], right_elbow[1:], right_wrist[1:])
                angle_l = calculate_angle(left_shoulder[1:], left_elbow[1:], left_wrist[1:])

                # Display angles on the live feed
                cv2.putText(image, str(int(angle_r)), tuple(right_elbow[1:]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(angle_l)), tuple(left_elbow[1:]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Display the live feed with detected pose and angles
            cv2.imshow('Arm Angle Analyzer', image)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Release the video object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()