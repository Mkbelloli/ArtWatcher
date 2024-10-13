
import os
import time

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-11.2-v8.1\cuda\bin')

import cv2
from ProcessorEngine import ProcessorEngine

CAMERA_INPUT = False
INPUT_VIDEO = 'videos\\Venice-2-raw.webm'


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Starting Computer Vision Project...')

    # Open the default camera
    if CAMERA_INPUT:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(INPUT_VIDEO)

    engine = ProcessorEngine()

    # Get the default frame width and height
    #frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while CAMERA_INPUT or cam.isOpened():
        ret, frame = cam.read()

        out_frame = engine.process_frame(frame)

        #out.write(frame)
        # Display the captured frame
        cv2.imshow('Camera', out_frame)
        cv2.setMouseCallback('Camera', click_event)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('p'):
            time.sleep(5)

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()

