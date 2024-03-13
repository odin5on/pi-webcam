print('importing cv2')
import cv2
print('import successful')

cap = cv2.VideoCapture(0)  # Open video capture object

while True:
    ret, frame = cap.read()  # Read one frame from the video capture object
    if not ret:
        break

    # Rotate the frame by 180 degrees
    frame = cv2.blur(frame)

    #cv2.imshow('Rotated Video', frame)  # Display the result
    cap.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
