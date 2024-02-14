import numpy as np
from picamera2 import Picamera2, H264Encoder
import time

class RotatingEncoder(H264Encoder):
    def input(self, frame):
        # Rotate the frame by 180 degrees
        frame = np.rot90(frame, 2)
        super().input(frame)

picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

encoder = RotatingEncoder(10000000)
picam2.start_recording(encoder, 'test.h264')

time.sleep(5)

picam2.stop_recording()

print('done')