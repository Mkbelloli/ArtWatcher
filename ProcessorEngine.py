
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
        self.__map_points = []

    def load_map(self, points):
        self.__map_points = points

    def __draw_map_on_frame(self, frame):
        for pi in range(len(self.__map_points)-1, -1, -1):
            cv2.line(frame, self.__map_points[pi], self.__map_points[pi-1], MAP_LINE_COLOR, 1)

        return frame

    def __draw_map_anchors(self, frame):
        for anch_p in self.__map_points:
            cv2.circle(frame, center=anch_p, radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)

        return frame

    def __show_people(self, frame, outputs):
        H, W = frame.shape[:2]

        boxes = []
        confidences = []
        classIDs = []
        map_polygon = np.array(self.__map_points, dtype=np.int32)

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)

            if classID != PERSON_CLASSID:
                continue

            if scores[classID] < self.__min_confidence_person:
                continue

            # draw person box
            x, y, w, h = output[:4] * np.array([W, H, W, H])

            if cv2.pointPolygonTest(map_polygon, (int(x), int(y+h/2)), False) < 0:
                continue

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
                cv2.circle(frame, (int(x+w/2), int(y+h)), 5, PERSON_POINT_COLOR, cv2.FILLED)
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

        people, frame = self.get_people(frame)

        return frame