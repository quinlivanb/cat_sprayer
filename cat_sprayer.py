from threading import Thread
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import config
from utils import ImageWrangler, VariableFifo, SprayController, MockController


def main():
    # initialize sprayer control
    if config.demo_mode:
        controller = MockController()
    else:
        controller = SprayController(config.pi_pin, config.spray_duration)

    cat_index = config.cat_index                                # output logit index corresponding to cat class
    min_conf_threshold = config.min_conf_threshold              # confidence level threshold

    # load model from disc
    model_path = config.model_loc
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # set up the video capture using open cv
    image_capture = ImageWrangler((width, height), cam_number=0)

    # set up buffer control
    expected_fps = config.expected_fps

    # initialize variable length fixed duration fifos
    variable_fifo = VariableFifo(initial_fps=expected_fps)
    detection_buffer = variable_fifo.initialize_buffers(False, config.detection_dur)
    frame_buffer = variable_fifo.initialize_buffers(None, config.video_dur)
    fps_buffer = variable_fifo.initialize_buffers(expected_fps, config.fps_dur)

    # flag and counter are used to ensure we only launch one thread per event
    event_frame_cnt = 0
    on_going_event = False

    # how full should the detection buffer be to trigger an event?
    detection_ratio = config.detection_ratio

    # frame_rate_buffer
    freq = cv2.getTickFrequency()
    cur_fpr = prev_fps = expected_fps

    # how many frames to wait before storing video buffer to disk
    capture_delay = max(1, frame_buffer.maxlen + (config.event_lead * cur_fpr))

    # initialize required flags
    sprayer_thread = None
    video_thread = None
    recording = False

    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        cur_fps = int(np.mean(fps_buffer))

        # Capture and pre_process latest available frame - also returns frame for display purposes
        input_data, cur_frame = image_capture.collect_frame()

        # Run inference of processed frame
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        # boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # currently unused
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        # was any cat observed in the current frame
        cat_detected = np.sum((classes == cat_index) & (scores > min_conf_threshold)) > 0

        # has the number of cat detections in the last n frames exceeded the detection ratio ratio?
        event_detected = sum(detection_buffer) / detection_buffer.maxlen > detection_ratio

        # start new event unless one on-going
        if event_detected and not on_going_event:
            on_going_event = True

            # launch tread to control sprayer without blocking video capture
            sprayer_thread = Thread(target=controller.spray_the_cat)
            sprayer_thread.start()

        # during an event, we use a timer to collect to correct number of frames and write them to file
        if on_going_event:
            # cnt down timer
            event_frame_cnt += 1

            cv2.putText(cur_frame, 'Transmitting video in t-minus %i seconds' %
                        ((capture_delay - event_frame_cnt) / expected_fps),
                        (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

            if sprayer_thread.is_alive():
                cv2.putText(cur_frame, 'Cat detected!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.putText(cur_frame, 'Activating cat deterrent devices',
                            (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            if event_frame_cnt >= capture_delay:
                on_going_event = False
                event_frame_cnt = 0
                # launch tread to store latest video clip
                video_thread = Thread(target=image_capture.store_clip, args=(frame_buffer.copy(), cur_fps,))
                video_thread.start()
                recording = True

        if recording and video_thread.is_alive():
            recording = False

        # Display frame for demo mode
        if config.demo_mode:
            cv2.imshow('Object detector', cur_frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        fps = 1 / ((t2 - t1) / freq)

        # update FIFO buffers
        frame_buffer.append(cur_frame)  # update frame FIFO buffer
        fps_buffer.appendleft(fps)  # update fps buffer
        detection_buffer.append(cat_detected)  # update event detection buffer

        # dynamically resize buffers to match current variable fps - ensure predicable algo behaviour on any device
        # don't update after event detection until video capture is complete
        if cur_fps != prev_fps and not recording:
            detection_buffer = variable_fifo.dynamic_buffer_update(config.detection_dur, detection_buffer, cur_fps)
            frame_buffer = variable_fifo.dynamic_buffer_update(config.video_dur, frame_buffer, cur_fps)
            fps_buffer = variable_fifo.dynamic_buffer_update(config.fps_dur, fps_buffer, cur_fps)
            # update capture delay to match new buffer sizes
            capture_delay = max(1, frame_buffer.maxlen + (config.event_lead * cur_fpr))
            # update prev_fps
            prev_fps = cur_fps

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and clean up the controller
    image_capture.tear_down()
    controller.tear_down()


if __name__ == '__main__':
    main()
