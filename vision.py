import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand module with adjustable confidence levels
min_detection_confidence = 0.7
min_tracking_confidence = 0.7

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

# MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Get Screen Resolution for Mouse Movement
screen_width, screen_height = pyautogui.size()

# Initialize previous positions for smoothing the mouse movement
prev_index_x, prev_index_y = 0, 0

# Variables to handle double-click gesture
last_click_time = 0
click_threshold = 0.5  # seconds between clicks to count as a double-click

# Set font for GUI feedback
font = cv2.FONT_HERSHEY_SIMPLEX

# Define a function to check double-click based on the time difference
def is_double_click():
    global last_click_time
    current_time = time.time()
    if current_time - last_click_time < click_threshold:
        last_click_time = 0  # Reset after double click is detected
        return True
    last_click_time = current_time
    return False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hand landmarks
    output = hands.process(rgb_frame)
    
    # Initialize feedback message
    feedback_message = ""

    # Check if hands are detected
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the list of landmarks
            landmarks = hand_landmarks.landmark
            
            # Initialize variables to store the positions of the thumb and index finger
            index_x, index_y = 0, 0
            thumb_x, thumb_y = 0, 0

            for id, landmark in enumerate(landmarks):
                # Get the (x, y) coordinates of each landmark
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw circles on the index finger (id == 8) and thumb (id == 4)
                if id == 8:
                    index_x, index_y = x, y
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # Index finger (yellow)

                if id == 4:
                    thumb_x, thumb_y = x, y
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # Thumb (blue)

            # Gesture detection logic
            # 1. Click Gesture (Index and Thumb close together)
            if abs(index_y - thumb_y) < 40:
                if is_double_click():
                    pyautogui.doubleClick()
                    feedback_message = "Double Click Detected"
                else:
                    pyautogui.click()
                    feedback_message = "Click Gesture Detected"
                pyautogui.sleep(1)  # Add a small delay to prevent multiple clicks

            # 2. Move Gesture (Index finger movement)
            elif abs(index_y - thumb_y) > 40:
                # Scale index finger position to screen resolution
                scaled_index_x = screen_width / frame.shape[1] * index_x
                scaled_index_y = screen_height / frame.shape[0] * index_y
                
                # Smooth the movement of the cursor
                smooth_factor = 0.3
                pyautogui.moveTo(
                    prev_index_x + (scaled_index_x - prev_index_x) * smooth_factor,
                    prev_index_y + (scaled_index_y - prev_index_y) * smooth_factor
                )

                # Update the previous index position
                prev_index_x, prev_index_y = scaled_index_x, scaled_index_y
                feedback_message = "Move Gesture Detected"

            # 3. Scroll Gesture (Detect vertical movement of the index finger)
            scroll_threshold = 30  # The threshold for scroll movement (in pixels)

            # Calculate the movement of the index finger (vertical movement only)
            vertical_move = index_y - prev_index_y

            # Scroll up if the index finger moves up (positive vertical move)
            if vertical_move > scroll_threshold:
                pyautogui.scroll(10)  # Scroll up
                feedback_message = "Scroll Up Gesture Detected"
            # Scroll down if the index finger moves down (negative vertical move)
            elif vertical_move < -scroll_threshold:
                pyautogui.scroll(-10)  # Scroll down
                feedback_message = "Scroll Down Gesture Detected"

            # 4. Right-click Gesture (Index and Thumb make a fist-like gesture)
            if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                pyautogui.rightClick()
                feedback_message = "Right-click Gesture Detected"

            # Display feedback message on the screen
            cv2.putText(frame, feedback_message, (10, 40), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with drawn landmarks and feedback
    cv2.imshow('Virtual Mouse', frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
