import cv2
import mediapipe as mp
import numpy as np


def check_index_fingers_crossed(mp_hands, hands_landmarks):
    # Assuming hands_landmarks contains landmarks for both hands
    if len(hands_landmarks) == 2:
        # Assuming the first hand in the list is the left hand and the second is the right hand
        # This assumption may not always hold true; you might need additional logic to confirm hand orientation

        # Get the first (MCP) and second (PIP) knuckle landmarks for the index fingers of both hands
        left_index_first_knuckle = hands_landmarks[1].landmark[
            mp_hands.HandLandmark.INDEX_FINGER_MCP
        ]
        right_index_first_knuckle = hands_landmarks[0].landmark[
            mp_hands.HandLandmark.INDEX_FINGER_MCP
        ]

        left_index_second_knuckle = hands_landmarks[1].landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
        right_index_second_knuckle = hands_landmarks[0].landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]

        # Check the conditions for "crossing" based on x-coordinates
        if (left_index_first_knuckle.x < right_index_first_knuckle.x) and (
            left_index_second_knuckle.x > right_index_second_knuckle.x
        ):
            return True
    return False


# Function to check thumb position relative to other fingers
#   This function compares the vertical position of thumb to the
#   vertical position of fingers

# Note: Lower values means higher in the frame (origin in top-left corner)


def check_thumb_position(mp_hands, hand_landmarks, comparison):

    # Extracts vertical position of thumb
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

    # Extracts vertical position of remaining fingers
    finger_tips_y = [
        hand_landmarks.landmark[finger_tip].y
        for finger_tip in [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP,
        ]
    ]

    # compare vertical height of thumb and finger with min height (highest in frame)- returns boolean
    return comparison(thumb_tip, min(finger_tips_y))


# Function to zoom image in (zoom_factor indicated how much the image should be zoomed in)
def zoom_image(image, zoom_factor):
    if zoom_factor == 1.0:  # No zoom needed
        return image

    # Extract the original height and width of the image
    height, width = image.shape[:2]

    # Compute new dimensions (zoom in if ZF > 1, zoom out if ZF < 1)
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Rescale image to new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # calculates top-left coordinates - ensures final image has same dimensions as original
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2

    # crops image while keeping original dimensions - this step is what creates the zooming effect
    cropped_image = resized_image[start_y : start_y + height, start_x : start_x + width]
    return cropped_image


def main():
    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils
    # Initialize webcam capture.
    cap = cv2.VideoCapture(0)

    camera_on = True  # Initial state

    # Zoom factor control
    current_zoom_factor = 1.0  # Start with no zoom
    frame_counter = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        zoom_changed = False
        
        if frame_counter % 5 == 0:
            if camera_on:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resultsHand = hands.process(image)

                if resultsHand.multi_hand_landmarks:
                    for hand_landmarks in resultsHand.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        if check_index_fingers_crossed(
                            mp_hands, resultsHand.multi_hand_landmarks
                        ):
                            camera_on = False  # "Turn off" the camera

                        # Check for thumbs-up (zoom in)
                        elif check_thumb_position(
                            mp_hands, hand_landmarks, lambda thumb, others: thumb < others
                        ):
                            current_zoom_factor *= 1.01  # Increase zoom by 1%
                            zoom_changed = True

                        # Check for thumbs-down (zoom out)
                        elif check_thumb_position(
                            mp_hands, hand_landmarks, lambda thumb, others: thumb > others
                        ):
                            current_zoom_factor /= 1.01  # Decrease zoom by 1%
                            zoom_changed = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            else:
                # Camera is "off": display a black screen with your name
                image = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a black image
                cv2.putText(
                    image,
                    "80",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            frame_counter = 0

        frame_counter += 1
        # Apply the current zoom factor, if changed
        # if zoom_changed:
        current_zoom_factor = max(
            1.0, min(current_zoom_factor, 3.0)
        )  # Limit zoom factor range for practicality
        image = zoom_image(image, current_zoom_factor)

        cv2.imshow("MediaPipe Hands", image)

        # Press escape to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break
        elif cv2.waitKey(5) & 0xFF == ord("c"):
            camera_on = True  # Toggle camera back on

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
