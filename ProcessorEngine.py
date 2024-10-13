
import cv2
import numpy as np

PERSON_CLASSID = 0
PERSON_BOX_COLOR = (150,150,0)
PERSON_POINT_COLOR = (250,250, 0)
MAP_LINE_COLOR = (0, 0, 250)
MAP_ANCHOR_COLOR = (0, 0, 250)
class ProcessorEngine:

    def __init__(self, SHOW_PEOPLE_BOX= True, PERSON_MIN_CONFIDENCE=0.7):
        self.__show_people_box = SHOW_PEOPLE_BOX
        self.__min_confidence_person = PERSON_MIN_CONFIDENCE

        self.__net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        #self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __draw_map_on_frame(self, frame):
        cv2.line(frame, (118, 287), (790, 287), MAP_LINE_COLOR, 1)
        cv2.line(frame, (5, 317), (118, 287), MAP_LINE_COLOR, 1)
        cv2.line(frame, (957, 334), (790, 287), MAP_LINE_COLOR, 1)

        cv2.line(frame, (5, 535), (955, 535), MAP_LINE_COLOR, 1)
        cv2.line(frame, (957, 334), (955, 535), MAP_LINE_COLOR, 1)
        cv2.line(frame, (5, 317), (5, 535), MAP_LINE_COLOR, 1)

        return frame

    def __draw_map_anchors(self, frame):
        cv2.circle(frame, center=(118, 287), radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)
        cv2.circle(frame, center=(790, 287), radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)
        cv2.circle(frame, center=(5, 317), radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)
        cv2.circle(frame, center=(957, 334), radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)

        cv2.circle(frame, center=(5, 535), radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)
        cv2.circle(frame, center=(955, 535), radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)

        return frame

    def __show_people(self, frame, outputs):
        H, W = frame.shape[:2]

        print(H)
        print(W)

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
                cv2.circle(frame, (x+int(w/2), y+h), 5, PERSON_POINT_COLOR, cv2.FILLED)
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
            frame = self.__draw_map_on_frame(frame)
            frame = self.__draw_map_anchors(frame)
        return outputs, frame

    def process_frame(self, frame):

        # transformation

        # faccio una trasformazione
        people, frame = self.get_people(frame)

        return frame