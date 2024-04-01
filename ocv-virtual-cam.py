import os
import sys
import cv2
import fcntl
from v4l2 import (
    v4l2_format, VIDIOC_G_FMT, V4L2_BUF_TYPE_VIDEO_OUTPUT, V4L2_PIX_FMT_RGB24,
    V4L2_FIELD_NONE, VIDIOC_S_FMT, VIDIOC_EXPBUF
)
print("OpenCV version: "+cv2.__version__)

VID_WIDTH = 640
VID_HEIGHT = 480

VIDEO_IN = "/dev/video0"
VIDEO_OUT = "/dev/video2"

def main():
    cam = cv2.VideoCapture("libcamerasrc ! appsink")
    if not cam.isOpened():
        print("ERROR: could not open camera!")
        return -1

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, VID_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_HEIGHT)

    try:
        output = os.open(VIDEO_OUT, os.O_RDWR)
    except Exception as ex:
        print("ERROR: could not open output device!")
        print(str(ex))
        return -1

    vid_format = v4l2_format()
    vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT

    if fcntl.ioctl(output, VIDIOC_G_FMT, vid_format) < 0:
        print("ERROR: unable to get video format!")
        return -1

    framesize = VID_WIDTH * VID_HEIGHT * 3
    vid_format.fmt.pix.width = VID_WIDTH
    vid_format.fmt.pix.height = VID_HEIGHT

    vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24
    vid_format.fmt.pix.sizeimage = framesize
    vid_format.fmt.pix.field = V4L2_FIELD_NONE

    if fcntl.ioctl(output, VIDIOC_S_FMT, vid_format) < 0:
        print("ERROR: unable to set video format!")
        return -1

    if(fcntl.ioctl(output, VIDIOC_EXPBUF, vid_format) < 0):
        print("ERROR: unable to export buffer!")
        return -1

    while True:
        ret, frame = cam.read()  # Read frame from physical camera
        print("read some data")
        if not ret:
            print("Error: Failed to capture frame")
            break

        result = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        written = os.write(output, result.data)
        if written < 0:
            print("ERROR: could not write to output device!")
            os.close(output)
            break

        # wait for user to finish program pressing ESC
        if cv2.waitKey(10) == 27:
            break

    print("\n\nFinish, bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

