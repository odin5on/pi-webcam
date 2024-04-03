import fcntl
import cv2
print("OpenCV version: "+cv2.__version__)
print(cv2.getBuildInformation())

# Open the physical camera
#physical_camera = cv2.VideoCapture("libcamerasrc ! capsfilter caps=video/x-raw,framerate=30/1 ! v4l2convert ! appsink")
physical_camera = cv2.VideoCapture("libcamerasrc ! appsink")
print("Physical camera opened successfully")

# Open the virtual camera device
virtual_camera = cv2.VideoWriter("appsrc ! videoconvert ! v4l2sink device=/dev/video3", -1, 30, (1920,1080),True)
print("Virtual camera opened successfully")

#if not virtual_camera.isOpened():
#    print("Error: Could not open virtual camera")
#    exit()


while True:
    ret, frame = physical_camera.read()  # Read frame from physical camera
    if not ret:
        print("Error: Failed to capture frame")
        break

    cv2.blur(frame, (5, 5))
    print("Frame processed")

    # Write the processed frame to the virtual camera
    virtual_camera.write(frame)
    print("Frame written to virtual camera")

    #cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
physical_camera.release()
virtual_camera.release()
cv2.destroyAllWindows()

