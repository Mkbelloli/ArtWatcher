

import cv2

CAMERA_INPUT = False
INPUT_VIDEO = 'output.mp4'

def process_frame(frame):
    o = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return o
"""
 people identification and segmentation
 position per person
 map calibration
 send position to the map
"""


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Starting Computer Vision Project...')

    # Open the default camera
    if CAMERA_INPUT:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(INPUT_VIDEO)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while CAMERA_INPUT or cam.isOpened():
        ret, frame = cam.read()

        out_frame = process_frame(frame)

        #out.write(frame)
        # Display the captured frame
        cv2.imshow('Camera', out_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()

