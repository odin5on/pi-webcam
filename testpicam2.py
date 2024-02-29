from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time
import libcamera


picam2 = Picamera2()
video_config = picam2.create_video_configuration()
video_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picam2.configure(video_config)

print(picam2.camera_controls)

encoder = H264Encoder(10000000)
picam2.start_recording(encoder, 'test.h264')

time.sleep(5)

picam2.stop_recording()

print('done')
