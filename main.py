import cv2
import mediapipe as mp

import math

def calculate_angle(finger_tip, finger_base, palm):
    # Calculate the vectors for finger_tip to finger_base and finger_tip to palm
    vec1 = [finger_tip.x - finger_base.x, finger_tip.y - finger_base.y]
    vec2 = [finger_tip.x - palm.x, finger_tip.y - palm.y]

    # Calculate the angle between the two vectors
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude_product = math.sqrt(vec1[0]**2 + vec1[1]**2) * math.sqrt(vec2[0]**2 + vec2[1]**2)
    angle_rad = math.acos(dot_product / magnitude_product)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def count_fingers(landmarks):
    # Indices of finger tip landmarks in the hand landmarks list
    tip_indices = [4, 8, 12, 16, 20]

    # Count the number of fingers that are "up"
    finger_count = 0

    # Check the angle between adjacent fingers
    for i in range(1, len(tip_indices)):
        finger_tip = landmarks[tip_indices[i]]
        finger_base = landmarks[tip_indices[i] - 2]
        palm = landmarks[0]  # Landmark for the center of the palm

        # Calculate the angle between the finger tip, finger base, and palm
        angle = calculate_angle(finger_tip, finger_base, palm)

        # Use a threshold angle to determine if the finger is raised or not
        # Adjust this threshold as needed based on your hand and camera setup
        threshold_angle = 100  # You can try different values here
        if angle < threshold_angle:
            finger_count += 1

    return finger_count
def display_camera_with_finger_detection():
    # Load Mediapipe hand tracking module
    mp_hands = mp.solutions.hands

    # Open the default camera (usually the first camera in the system)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Initialize Mediapipe hand tracking
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Convert the frame from BGR to RGB format (required by Mediapipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe hand tracking
            results = hands.process(rgb_frame)

            # Check if hand(s) are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    # Count fingers and display the count
                    finger_count = count_fingers(hand_landmarks.landmark)
                    cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with finger detection
            cv2.imshow("Camera Footage with Finger Detection", frame)

            # Wait for 1 millisecond and check if 'q' key is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_finger_detection()
