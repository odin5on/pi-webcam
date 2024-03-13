print("importing cv2")
import cv2
print('import successful')
# Open the virtual camera device
virtual_camera = cv2.VideoCapture('/dev/video3')  # Adjust the path as per your system
print("Virtual camera opened successfully")

# Check if the virtual camera is opened successfully
if not virtual_camera.isOpened():
    print("Error: Could not open virtual camera")
    exit()

# Open the physical camera or any other video source
physical_camera = cv2.VideoCapture(0)  # Adjust index if you have multiple cameras
print("Physical camera opened successfully")

while True:
    ret, frame = physical_camera.read()  # Read frame from physical camera
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Perform image processing operations on the frame
    cv2.blur(frame, (5, 5))
    print("Frame processed")
    # For example, you can use OpenCV functions like cv2.cvtColor(), cv2.blur(), etc.

    # Write the processed frame to the virtual camera
    virtual_camera.write(frame)
    print("Frame written to virtual camera")

    # Display the processed frame (optional)
    #cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
physical_camera.release()
virtual_camera.release()
cv2.destroyAllWindows()

