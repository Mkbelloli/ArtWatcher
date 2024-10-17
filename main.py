
import os
import time

import numpy as np
import utils

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-11.2-v8.1\cuda\bin')

import cv2
from ProcessorEngine import ProcessorEngine

CAMERA_INPUT = False
INPUT_VIDEO = 'videos\\Venice-2-raw.webm'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    utils.deactivate_logs()
    utils.print_log('Starting Computer Vision Project...')

    # Open the default camera
    if CAMERA_INPUT:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(INPUT_VIDEO)

    engine = ProcessorEngine()
    engine.load_map( np.array([(118, 287), (790, 287), (957, 334), (957, 535), (5, 535), (5, 317)  ]))
    engine.load_mapped_points( [[118, 287], [790, 287], [957, 334], [5, 317]],
                               [[0, 16], [16, 16], [16, 11.5], [0, 14]])

    while CAMERA_INPUT or cam.isOpened():
        ret, frame = cam.read()

        out_frame = engine.process_frame(frame)

        #out.write(frame)
        # Display the captured frame
        cv2.imshow('Camera', out_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(1) == ord('p'):
            time.sleep(5)

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()

