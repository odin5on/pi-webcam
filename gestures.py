import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection


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


def blur_background(image, face_bbox):
    # Create a mask with the same size as the image
    mask = np.zeros_like(image)

    # Draw a filled rectangle on the mask to represent the face region
    cv2.rectangle(
        mask,
        (face_bbox[0], face_bbox[1]),
        (face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]),
        (255, 255, 255),
        -1,
    )

    # Apply a blur filter to the entire image
    blurred_image = cv2.GaussianBlur(image, (99, 99), 0)

    # Use the mask to blend the original image with the blurred image
    blended_image = cv2.bitwise_and(image, mask) + cv2.bitwise_and(
        blurred_image, cv2.bitwise_not(mask)
    )

    return blended_image


def check_pointer_finger_position(mp_hands, hand_landmarks):
    # Get the y-coordinate of the pointer finger tip
    pointer_finger_tip_y = hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    ].y
    # Get the height of the screen
    screen_height = 720  # Assuming a screen height of 720 pixels
    # Check if the pointer finger is in the top half of the screen
    if (pointer_finger_tip_y * screen_height) < (screen_height / 6):
        return True
    return False


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
    blur_on = False  # Initial state

    # Zoom factor control
    current_zoom_factor = 1.0  # Start with no zoom
    frame_counter = 0

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detector:

        while cap.isOpened():

            if not camera_on:
                # Camera is "off": display a black screen with your name
                zoom_on_shifted_image = np.zeros(
                    (720, 1280, 3), dtype=np.uint8
                )  # Create a black image
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

            else:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                zoom_changed = False

                rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results_face = face_detector.process(rgb_frame)
                image_height, image_width, c = image.shape

                if results_face.detections:
                    for face in results_face.detections:
                        bboxC = face.location_data.relative_bounding_box
                        bbox = (
                            int(bboxC.xmin * image_width),
                            int(bboxC.ymin * image_height),
                            int(bboxC.width * image_width),
                            int(bboxC.height * image_height),
                        )

                        if blur_on:
                            image = blur_background(image, bbox)

                        center_x = bbox[0] + bbox[2] // 2
                        center_y = bbox[1] + bbox[3] // 2

                        # Calculate the shift needed to center the face
                        shift_x = image_width // 2 - center_x
                        shift_y = image_height // 2 - center_y
                        # print("shift_x", shift_x)
                        # print("shift_y", shift_y)

                        max_x_shift = (
                            image_width - (image_width // current_zoom_factor)
                        ) / 2
                        max_y_shift = (
                            image_height - (image_height // current_zoom_factor)
                        ) / 2
                        # print("max_x_shift", max_x_shift)
                        # print("max_y_shift", max_y_shift)

                        shift_x = min(abs(shift_x), max_x_shift) * np.sign(shift_x)
                        shift_y = min(abs(shift_y), max_y_shift) * np.sign(shift_y)
                        # print("final shift_x", shift_x)
                        # print("final shift_y", shift_y)

                        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                        shifted_image = cv2.warpAffine(
                            image, M, (image_width, image_height)
                        )

                zoom_on_shifted_image = zoom_image(shifted_image, current_zoom_factor)

                if frame_counter % 5 == 0:

                    resultsHand = hands.process(zoom_on_shifted_image)

                    if resultsHand.multi_hand_landmarks:
                        for hand_landmarks in resultsHand.multi_hand_landmarks:
                            pointer_finger_tip_y = (
                                hand_landmarks.landmark[
                                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                                ].y
                                * image_width
                            )
                            pointer_finger_tip_x = (
                                hand_landmarks.landmark[
                                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                                ].x
                                * image_height
                            )

                            # mp_draw.draw_landmarks(
                            #     shifted_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            # )

                            # if check_index_fingers_crossed(
                            #     mp_hands, resultsHand.multi_hand_landmarks
                            # ):
                            #     camera_on = False  # "Turn off" the camera

                            # # Check for thumbs-up (zoom in)
                            # elif check_thumb_position(
                            #     mp_hands,
                            #     hand_landmarks,
                            #     lambda thumb, others: thumb < others,
                            # ):
                            #     current_zoom_factor *= 1.05  # Increase zoom by 1%
                            #     zoom_changed = True

                            # # Check for thumbs-down (zoom out)
                            # elif check_thumb_position(
                            #     mp_hands,
                            #     hand_landmarks,
                            #     lambda thumb, others: thumb > others,
                            # ):
                            #     current_zoom_factor /= 1.05  # Decrease zoom by 1%
                            #     zoom_changed = True

                            # if check_pointer_finger_position(mp_hands, hand_landmarks):
                            #     print("Pointer finger is in the top fourth of the screen")

                            if pointer_finger_tip_y < image_width / 6:
                                print(
                                    "Pointer finger is in the top sixth of the screen"
                                )
                                if pointer_finger_tip_x < image_height / 6:
                                    print("Pointer finger is in the leftmost part of the screen")
                                    current_zoom_factor *= 1.05
                                    zoom_changed = True
                                elif pointer_finger_tip_x < 2 * image_height / 6:
                                    print("Pointer finger is in the second part of the screen")
                                    current_zoom_factor /= 1.05
                                    zoom_changed = True
                                elif pointer_finger_tip_x < 3 * image_height / 6:
                                    print("Pointer finger is in the third part of the screen")
                                    
                                elif pointer_finger_tip_x < 4 * image_height / 6:
                                    print("Pointer finger is in the fourth part of the screen")
                                    
                                elif pointer_finger_tip_x < 5 * image_height / 6:
                                    print("Pointer finger is in the fifth part of the screen")
                                    
                                else:
                                    print("Pointer finger is in the rightmost part of the screen")
                                    

                    frame_counter = 0

                frame_counter += 1
                # Apply the current zoom factor, if changed
                if zoom_changed:
                    current_zoom_factor = max(
                        1.0, min(current_zoom_factor, 5.0)
                    )  # Limit zoom factor range for practicality

                # Draw a box around the top 6th part of the screen

            
            for i in range(6):
                cv2.rectangle(
                    zoom_on_shifted_image,
                    (i * (image_width // 6), 0),
                    ((i + 1) * (image_width // 6), image_height // 6),
                    (0, 255, 0),
                    2,
                )

            # Draw a plus sign in the leftmost rectangle
            cv2.putText(
                zoom_on_shifted_image,
                "+",
                (image_width // 12, image_height // 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw a minus sign in the second left rectangle
            cv2.putText(
                zoom_on_shifted_image,
                "-",
                (3 * image_width // 12, image_height // 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("MediaPipe Hands", zoom_on_shifted_image)

            # print("Dimensions of zoom_on_shifted_image:", zoom_on_shifted_image.shape)

            # Press escape to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
            elif cv2.waitKey(5) & 0xFF == ord("c"):
                camera_on = True  # Toggle camera back on
            elif cv2.waitKey(5) & 0xFF == ord("b"):
                blur_on = not blur_on

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
