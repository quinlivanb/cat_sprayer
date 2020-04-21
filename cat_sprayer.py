from threading import Thread
import time
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import config
from utils import ImageWrangler, VariableFifo, SprayController, MockController, SqlControl


def main(controller, image_capture, sql_interface):

    # initialize variable length fixed duration fifos
    variable_fifo = VariableFifo(initial_fps=config.expected_fps)
    detection_buffer = variable_fifo.initialize_buffers(False, config.detection_dur)
    frame_buffer = variable_fifo.initialize_buffers(None, config.video_dur)
    fps_buffer = variable_fifo.initialize_buffers(config.expected_fps, config.fps_dur)

    # frame_rate_buffer
    freq = cv2.getTickFrequency()
    cur_fpr = prev_fps = config.expected_fps

    # how many frames to wait before storing video buffer to disk
    capture_delay = max(1, frame_buffer.maxlen + (config.event_lead * cur_fpr))

    # initialize required loop variables
    sprayer_thread = None
    event_frame_cnt = 0
    on_going_event = False
    cur_event_start = 0

    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        cur_fps = int(np.mean(fps_buffer))

        # Capture and pre_process latest available frame - also returns frame for display purposes
        input_data, cur_frame = image_capture.collect_frame()

        if cur_frame is None:
            print('passing null frame')
            continue

        # Run inference of processed frame
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        # boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # currently unused
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        # was any cat observed in the current frame
        cat_detected = np.sum((classes == config.cat_index) & (scores > config.min_conf_threshold)) > 0

        # has the number of cat detections in the last n frames exceeded the detection ratio ratio?
        event_detected = sum(detection_buffer) / detection_buffer.maxlen > config.detection_ratio

        # start new event unless one on-going
        if event_detected and not on_going_event:
            print('Event Detected')
            on_going_event = True
            cur_event_start = time.time()

            # log event to SQL DB
            sql_interface.insert_event()

            # launch tread to control sprayer without blocking video capture
            if config.sprayer_engaged:
                sprayer_thread = Thread(target=controller.spray_the_cat)
                sprayer_thread.start()

        # during an event, we use a timer to collect to correct number of frames and write them to file
        if on_going_event:
            # cnt down timer
            event_frame_cnt += 1

            # add text to current frame
            if config.add_text:
                cur_frame = image_capture.draw_text(cur_frame, cur_event_start, sprayer_thread.is_alive())

            if event_frame_cnt >= capture_delay:
                on_going_event = False
                event_frame_cnt = 0
                # launch tread to store latest video clip
                if config.send_clip:
                    video_thread = Thread(target=image_capture.store_clip, args=(frame_buffer.copy(), cur_fps,))
                    video_thread.start()

        # Display frame for demo mode
        if config.show_vid and not config.on_pi:
            cv2.imshow('Object detector', cur_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Calculate framerate
        t2 = cv2.getTickCount()
        fps = 1 / ((t2 - t1) / freq)

        # update FIFO buffers
        frame_buffer.append(cur_frame)  # update frame FIFO buffer
        fps_buffer.appendleft(fps)  # update fps buffer
        detection_buffer.append(cat_detected)  # update event detection buffer

        # dynamically resize buffers to match current variable fps - ensure predicable algo behaviour at any fps
        # don't update after event detection until video capture is complete
        if cur_fps != prev_fps:
            detection_buffer = variable_fifo.dynamic_buffer_update(config.detection_dur, detection_buffer, cur_fps)
            frame_buffer = variable_fifo.dynamic_buffer_update(config.video_dur, frame_buffer, cur_fps)
            fps_buffer = variable_fifo.dynamic_buffer_update(config.fps_dur, fps_buffer, cur_fps)
            # update capture delay to match new buffer sizes
            capture_delay = max(1, frame_buffer.maxlen + (config.event_lead * cur_fpr))
            # update prev_fps
            prev_fps = cur_fps


if __name__ == '__main__':

    # initialize sprayer control
    if config.on_pi:
        hardware_controller = SprayController(config.pi_pin, config.spray_duration)
    else:
        hardware_controller = MockController()

    # SQL interface
    sql_interface = SqlControl()
    sql_interface.create_table()

    # load model from disc
    interpreter = Interpreter(model_path=config.model_loc)
    interpreter.allocate_tensors()

    # get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # set up the video capture using open cv
    opencv_capture = ImageWrangler((width, height), cam_number=0)

    try:
        main(hardware_controller, opencv_capture, sql_interface)
    except KeyboardInterrupt:
        # When everything done, release the capture and clean up the controller
        opencv_capture.tear_down()
        hardware_controller.tear_down()
