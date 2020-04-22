from collections import deque
import time
from datetime import datetime
import os
import subprocess
import shutil
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import boto3
from botocore.exceptions import ClientError
import config
import sqlite3
from sqlite3 import Error


if config.on_pi:
    import RPi.GPIO as GPIO


class SprayController:
    def __init__(self, target_pin, spray_duration):
        self.target_pin = target_pin
        self.spray_duration = spray_duration
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(target_pin, GPIO.OUT)

    def spray_the_cat(self):
        # start.sh the sprayer with 2 second long pause
        GPIO.output(self.target_pin, GPIO.HIGH)
        # spray the cat for 3 seconds
        time.sleep(self.spray_duration)
        # turn off the sprayer
        GPIO.output(self.target_pin, GPIO.LOW)

    def tear_down(self):
        # ensure target pin tied to low
        GPIO.output(self.target_pin, GPIO.LOW)


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
        cam = cv2.VideoCapture(cam_number)
        cam.release()
        self.capture = cv2.VideoCapture(cam_number)
        self.capture.set(3, config.resolution[0])
        self.capture.set(4, config.resolution[1])
        time.sleep(config.cam_warm_up)
        self.input_dims = input_dims
        self.send_clip = SendClip()
        self.temp_video_dir = config.video_out
        # create temp folder, will be deleted at tear down
        if not os.path.exists(self.temp_video_dir.split('/')[0]):
            os.mkdir(self.temp_video_dir.split('/')[0])

    def collect_frame(self):
        # Capture latest available frame
        _, frame = self.capture.read()

        # Prepare input frame for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.input_dims)
        np_data = np.expand_dims(frame_resized, axis=0)

        return np_data, frame

    def store_clip(self, frames, fps):
        (h, w) = reversed(config.resolution)

        temp_file = self.temp_video_dir + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '_temp' + config.clip_ext
        output_clip = self.temp_video_dir + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + config.clip_ext

        out = cv2.VideoWriter(temp_file, config.codec, fps, (w, h))
        for frame in frames:
            out.write(cv2.resize(frame, (w, h)))
        out.release()

        # slight hack to covert file to h264, also need to add silent audio track to keep whatsapp happy!!
        fnull = open(os.devnull, 'w')
        subprocess.run(("ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100"
                       " -i %s -c:v libx264 -c:a aac -shortest %s" % (temp_file, output_clip)).split(' '),
                       stdout=fnull, stderr=subprocess.STDOUT)

        self.send_clip.send_mms(output_clip)

        # delete local files when no longer needed - not much space on the pi!
        self.delete_file(temp_file)
        self.delete_file(output_clip)

    @staticmethod
    def draw_text(frame, event_time, sprayer_thread):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                  # Convert the image to RGB (OpenCV uses BGR)
        frame_pil = Image.fromarray(frame_rgb)                              # Pass the image to PIL
        draw = ImageDraw.Draw(frame_pil)
        font = ImageFont.truetype(config.font, config.text_size)            # setup font

        dots = int((time.time() - event_time)/2)
        draw.text((50, frame.shape[0] - 50), 'Preparing video for transmission.' + '.' * dots,
                  font=font, fill=config.green)

        if sprayer_thread:
            draw.text((50, 50), 'Cat detected!', font=font, fill=config.red)
            draw.text((50, 100), 'Activating cat deterrent devices', font=font, fill=config.green)

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    @staticmethod
    def delete_file(file_name):
        if os.path.exists(file_name):
            os.remove(file_name)

    def tear_down(self):
        self.capture.release()
        cv2.destroyAllWindows()
        # delete temp folder and any remaining videos
        shutil.rmtree(self.temp_video_dir.split('/')[0])


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


class SqlControl:
    def __init__(self):
        self.db_location = config.event_db

    def create_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_location, isolation_level=None)
        except Error as e:
            print(e)

        return conn

    def create_table(self):
        conn = self.create_connection()
        sql = "CREATE TABLE IF NOT EXISTS events (date_time datetime PRIMARY KEY)"
        try:
            c = conn.cursor()
            c.execute(sql)
        except Error as e:
            print(e)
        conn.close()

    def insert_event(self):
        conn = self.create_connection()
        sql = "INSERT INTO events(date_time) VALUES(datetime('now', 'localtime'))"
        cur = conn.cursor()
        cur.execute(sql)
        conn.close()
