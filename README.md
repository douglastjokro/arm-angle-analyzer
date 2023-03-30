# Arm Angle Analyzer
This program uses MediaPipe Pose to detect the pose of a person in a live video feed, and particularly calculates the angle between their right and left arms.

## Dependencies
- Python 3.x
- OpenCV
- Mediapipe
- Numpy

## Installation
- Clone the repo
- Install the dependencies using '**pip install -r requirements.txt**'

## Execution
1. Run the program with this command: python arm_angle_analyzer.py
2. This will open a window showing the live video feed from your default camera, with the pose landmarks and angles overlaid on the video.
3. The script will continue to run until you press the 'q' key to quit.

## Credits
This program was developed by Douglas Tjokrosetio and is based on the Mediapipe library from Google.

License
This project is licensed under the MIT License - see the LICENSE file for details.
