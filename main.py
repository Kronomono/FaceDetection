import cv2

def display_camera():
    # Open the default camera (usually the first camera in the system)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Display the frame
        cv2.imshow("Camera Footage", frame)

        # Wait for 1 millisecond and check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera()
