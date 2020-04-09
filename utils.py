from collections import deque
import time
from datetime import datetime
import os
import numpy as np
import cv2
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import boto3
from botocore.exceptions import ClientError
import config

if not config.demo_mode:
    import RPi.GPIO as GPIO


class SprayController:
    def __init__(self, target_pin, spray_duration):
        self.target_pin = target_pin
        self.spray_duration = spray_duration
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(target_pin, GPIO.OUT)

    def spray_the_cat(self):
        # start the sprayer with 2 second long pause
        GPIO.output(self.target_pin, GPIO.HIGH)
        time.sleep(config.on_press_duration)
        GPIO.output(self.target_pin, GPIO.LOW)

        # spray the cat for 3 seconds
        time.sleep(self.spray_duration)

        # turn off the sprayer
        GPIO.output(self.target_pin, GPIO.HIGH)
        time.sleep(config.off_press_duration)
        GPIO.output(self.target_pin, GPIO.LOW)

    @staticmethod
    def tear_down():
        GPIO.cleanup()


# this class is just used for demo mode
class MockController:
    @staticmethod
    def spray_the_cat():
        print('Spraying Cat')
        time.sleep(3)
        print('Cat has been tamed')

    @staticmethod
    def tear_down():
        pass


class SendClip:
    def __init__(self):
        s3 = boto3.resource('s3')
        self.s3_client = boto3.client('s3')
        self.bucket = s3.Bucket(config.bucket_name)
        self.twilio_client = Client(os.getenv('TWILIO_ID'), os.getenv('TWILIO_KEY'))

    def send_mms(self, file_name):
        # create media url using s3 pre-signed url
        media = self.generate_url(file_name)
        # send message using twilio
        self.send_twilio(media)

    def generate_url(self, file_name):
        # create media url using s3 pre-signed url
        try:
            self.bucket.upload_file(file_name, file_name.split('/')[-1],
                                    ExtraArgs={'ContentType': config.content_type})

            # generate temp pre-signed url using local aws credentials
            params = {'Bucket': config.bucket_name, 'Key': file_name.split('/')[-1]}
            media = self.s3_client.generate_presigned_url('get_object', Params=params, ExpiresIn=config.url_duration)
        except ClientError:
            media = None

        return media

    def send_twilio(self, media):
        # send message using twilio
        try:
            if media:
                self.twilio_client.messages.create(to=config.mms_target, from_=config.mms_source,
                                                   body=config.mms_text, media_url=media)
            else:
                self.twilio_client.messages.create(to=config.mms_target, from_=config.mms_source,
                                                   body=config.mms_text_media_fail)
        except TwilioRestException:
            pass


class ImageWrangler:
    def __init__(self, input_dims, cam_number=config.cam_number):
        self.capture = cv2.VideoCapture(cam_number)
        time.sleep(config.cam_warm_up)
        self.input_dims = input_dims
        self.send_clip = SendClip()

    def collect_frame(self):
        # Capture latest available frame
        _, frame = self.capture.read()

        # Prepare input frame for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.input_dims)
        np_data = np.expand_dims(frame_resized, axis=0)

        return np_data, frame

    def store_clip(self, frames, fps):
        # todo set image dimentions as config
        (h, w) = frames[0].shape[:2]

        # reduce resolution for mms due to 3MB data restriction
        if not config.use_whatsapp:
            h = int(h/2)
            w = int(h / 2)

        temp_file = config.video_out + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + config.clip_ext
        output_clip = config.video_out + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '_2' + config.clip_ext

        out = cv2.VideoWriter(temp_file, config.codec, fps, (w, h))
        for frame in frames:
            out.write(cv2.resize(frame, (w, h)))
        out.release()

        # slight hack to covert file to h264, also need to add silent audio track to keep whatsapp happy!!
        os.system("ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100"
                  " -i %s -c:v libx264 -c:a aac -shortest %s" % (temp_file, output_clip))

        self.send_clip.send_mms(output_clip)

        # delete local files when no longer needed - not much space on the pi!
        os.remove(temp_file)
        os.remove(output_clip)

    def tear_down(self):
        self.capture.release()
        cv2.destroyAllWindows()


# basically just deque but you can redefine the man len, if size is reduced, data is removed from the left
class VariableFifo:
    def __init__(self, initial_fps):
        self.initial_fps = initial_fps

    def initialize_buffers(self, fill_val, duration):
        buffer_size = self.initial_fps * duration
        if fill_val is None:
            buffer = deque(maxlen=buffer_size)
        else:
            buffer = deque([fill_val] * duration, maxlen=buffer_size)
        return buffer

    @staticmethod
    def dynamic_buffer_update(duration, buffer, cur_fps):
        buffer_list = list(buffer)
        buffer = deque(buffer_list, cur_fps * duration)
        return buffer
