import cv2
import os


# if not running on Raspberry pi
on_pi = True                                    # demo mode only sends video, no sprayer control available
show_vid = False

# algorithm settings
cat_index = 16                                  # logit index corresponding to cat class
model_loc = 'model_files/detect.tflite'         # location of quantized tflite model
min_conf_threshold = 0.65                       # confidence level threshold
detection_ratio = 0.25                          # how ful detection buffer must be to trigger event

# buffer setup
if on_pi:
    expected_fps = 4                            # starting point for dynamic fps buffers
else:
    expected_fps = 10                           # starting point for dynamic fps buffers

detection_dur = 1                               # buffer length in seconds
video_dur = 12                                  # duration (seconds) of output video
fps_dur = 10                                    # buffer length in seconds

# spray_control
sprayer_engaged = True                          # trigger sprayer
on_press_duration = 2.5                         # how long (seconds) to press button when turning on sprayer
off_press_duration = 0.5                        # how long (seconds) to press button when turning off sprayer
spray_duration = 1                              # how long (seconds) to spray the cat
pi_pin = 18                                     # control pin os raspberry pi

# sending clips
send_clip = True
bucket_name = 'cat-spray-clips'                 # s3 bucket to stores clips
use_whatsapp = False                            # send to whatsapp or stanard mms
if use_whatsapp:
    mms_target = os.getenv('WHATSAPP_TARGET')
    mms_source = os.getenv('WHATSAPP_SOURCE')
else:
    mms_target = os.getenv('MMS_TARGET')
    mms_source = os.getenv('MMS_SOURCE')

mms_text = "The cat is at it again..."
mms_text_media_fail = "The cat is at it again... " \
                      "but there was an issue with the media upload"
url_duration = 600

# camera control
resolution = (640, 480)
cam_number = 0                                  # index of camera
cam_warm_up = 2                                 # time (seconds) for camera sensor to 'warm up'
video_out = 'video_out/cat_clip_'               # where to store latest event video
codec = cv2.VideoWriter_fourcc(*'mp4v')         # video codec used by cv2
clip_ext = '.mp4'                               # base name for all video files
event_lead = -2                                 # when should clip start.sh, relative to event detection
content_type = 'video/mp4'                      # url content type


# video text
add_text = False
font = "fonts/UniVGA16.ttf"
text_size = 20
green = (0, 255, 43)
red = (217, 33, 33)

#
event_db = 'cat_sprayer_db.db'
