
import cv2
import numpy as np

PERSON_CLASSID = 0
PERSON_BOX_COLOR = (150,150,0)
class ProcessorEngine:

    def __init__(self, SHOW_PEOPLE_BOX= True, PERSON_MIN_CONFIDENCE=0.9):
        self.__show_people_box = SHOW_PEOPLE_BOX
        self.__min_confidence_person = PERSON_MIN_CONFIDENCE

        self.__net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        #self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __show_people(self, frame, outputs):
        H, W = frame.shape[:2]

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            if classID != PERSON_CLASSID:
                continue
            if scores[classID] < self.__min_confidence_person:
                continue

            # draw person box
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w // 2), int(y - h // 2)
            p1 = int(x + w // 2), int(y + h // 2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(scores[classID])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.__min_confidence_person, self.__min_confidence_person - 0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = PERSON_BOX_COLOR
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        return frame
    def get_people(self, frame):

        ln = self.__net.getLayerNames()
        ln = [ln[i - 1] for i in self.__net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256), swapRB=True, crop=False)
        self.__net.setInput(blob)
        outputs = self.__net.forward(ln)
        outputs = np.vstack(outputs)

        if self.__show_people_box:
            frame = self.__show_people(frame, outputs)

        return outputs, frame

    def process_frame(self, frame):

        # transformation

        # faccio una trasformazione
        people, frame = self.get_people(frame)

        return frame