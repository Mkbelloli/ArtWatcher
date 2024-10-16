import uuid
import cv2
import numpy as np
import socket

PERSON_CLASSID = 0
PERSON_BOX_COLOR = (150,150,0)
PERSON_POINT_COLOR = (250,250, 0)
MAP_LINE_COLOR = (0, 0, 250)
MAP_ANCHOR_COLOR = (0, 0, 250)

class ProcessorEngine:

    def __init__(self, SHOW_PEOPLE_BOX= True, PERSON_MIN_CONFIDENCE=0.7, track_on_map=True):
        self.__show_people_box = SHOW_PEOPLE_BOX
        self.__min_confidence_person = PERSON_MIN_CONFIDENCE
        self.__track_on_map = track_on_map

        self.__net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        #self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.__map_points = []
        self.__camera_map_points = []
        self.__real_map_points = []
        self.__H = None

        self.__curr_frame = None
        self.__prev_frame = None

        self.__client_socket = None
        if self.__track_on_map:
            self.__client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__client_socket.connect(('127.0.0.1', 65432))

        self.__people_dict = {}

    def __insert_person_info(self, name, map_point, box, key_points):
        self.__people_dict[name]= { 'name': name,
                                    'map_point': map_point,
                                    'box': box,
                                    'key_points': key_points,
                                    'missing_frame': 0}
        print(f"Inserito {name}")

    def __edit_person_info(self, name, map_point, box, key_points):
        self.__people_dict[name]= { 'name': name,
                                    'map_point': map_point,
                                    'box': box,
                                    'key_points': key_points,
                                    'missing_frame': 0}
        print(f"Aggiornato {name}")

    def __get_people_boxes(self):
        boxes = [x['box'] for x in  self.__people_dict.values()]
        return boxes

    def __get_people(self, query_box):

        if self.__prev_frame is None or \
            len(self.__people_dict)==0:
            print('Fase iniziale senza dati')
            return -1, None

        prev_gray = cv2.cvtColor(self.__prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(self.__curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Seleziona i punti di interesse all'interno del bounding box nel primo fotogramma
        x, y, w, h = query_box
        roi = prev_gray[int(y):int(y + h), int(x):int(x + w)]  # Region of Interest (ROI) per il box
        prev_points = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if prev_points is not None:
            # Aggiusta le coordinate dei punti rispetto all'intera immagine
            prev_points += np.array([[x, y]])
        else:
            # prev_points is None:
            print('prev_point a None')
            return -1, None  # Nessun punto da tracciare, restituisce None

        # 2. Traccia i punti nel secondo fotogramma usando l'Optical Flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None)

        # Filtra i punti validi
        good_prev = prev_points[status == 1]
        good_next = next_points[status == 1]

        # 3. Conta quanti punti finiscono all'interno di ciascuno dei box vicini nel secondo fotogramma
        counts = []
        people_boxes = self.__get_people_boxes()
        for next_box in people_boxes:
            x, y, w, h = next_box
            inside_box_count = 0
            for point in good_next:
                # Verifica se il punto tracciato è all'interno del bounding box
                if x <= point[0] <= x + w and y <= point[1] <= y + h:
                    inside_box_count += 1
            counts.append(inside_box_count)

        # 4. Trova il box con il maggior numero di punti tracciati dentro
        best_match_index = np.argmax(counts)
        best_match_obj = list(iter(self.__people_dict.values()))[best_match_index]
        best_match_count = counts[best_match_index]

        if best_match_count == 0:
            print('best match count a zero')
            return -1, None

        return best_match_count, best_match_obj

    def load_mapped_points(self, camera_points, real_points):
        self.__real_map_points = np.array(real_points)
        self.__camera_map_points = np.array(camera_points)
        self.__H, status = cv2.findHomography(self.__camera_map_points, self.__real_map_points)

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

    def __show_people(self, frame, people_boxes):
        for box in people_boxes:
            color = PERSON_BOX_COLOR
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), color, 1)
            cv2.circle(frame, (int(box[0]+box[2]/2), int(box[1]+box[3])), 5, PERSON_POINT_COLOR, cv2.FILLED)
        return frame

    def __trace_point(self, name, x, y):
        H = self.__H
        transformed_point = cv2.perspectiveTransform(np.array([[[x, y]]]), H)

        # Dividi per il terzo elemento (fattore di scala w) per ottenere le coordinate finali
        x_prime = transformed_point[0][0][0] #/ transformed_point[2]
        y_prime = transformed_point[0][0][1] #/ transformed_point[2]
        message = f"{name} {x_prime} {y_prime} <END>"
        print(f"Sent: {message}")
        self.__client_socket.send(message.encode('utf-8'))

    def get_people_from_frame(self, frame):
        H, W = frame.shape[:2]
        ln = self.__net.getLayerNames()
        ln = [ln[i - 1] for i in self.__net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.__net.setInput(blob)
        outputs = self.__net.forward(ln)
        outputs = np.vstack(outputs)
        boxes = []
        confidences = []
        map_polygon = np.array(self.__map_points, dtype=np.int32)
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            x = int(x-w/2)
            y = int(y-h/2)
            if classID != PERSON_CLASSID or \
               scores[classID] < self.__min_confidence_person or \
               cv2.pointPolygonTest(map_polygon, (int(x), int(y + h / 2)), False) < 0:
                continue

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(scores[classID])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.__min_confidence_person, self.__min_confidence_person - 0.1)
        people_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                # x,y,w,h
                people_boxes.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))
                """
                color = PERSON_BOX_COLOR
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (int(x + w / 2), int(y + h)), 5, PERSON_POINT_COLOR, cv2.FILLED)
"""
        return people_boxes, frame

    def process_frame(self, frame):
        self.__prev_frame = self.__curr_frame
        self.__curr_frame = frame # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        people_boxes, frame = self.get_people_from_frame(self.__curr_frame)

        if self.__prev_frame is None:
            return frame# jump first run, I need both

        prev_gray = cv2.cvtColor(self.__prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(self.__curr_frame, cv2.COLOR_BGR2GRAY)

        for pbox in people_boxes:
            score, people = self.__get_people(pbox)

            if people is None:
                x, y, w, h = pbox
                x = int(x - (w/2))
                y = int(y - (h/2))
                prev_gray = cv2.cvtColor(self.__curr_frame, cv2.COLOR_BGR2GRAY)
                key_points = cv2.goodFeaturesToTrack(prev_gray[int(y):int(y + h), int(x):int(x + w)], maxCorners=100, qualityLevel=0.3,
                                                      minDistance=7,
                                                      blockSize=7)

                if not key_points is None:
                    key_points += np.array([[int(x), int(y)]])

                self.__insert_person_info(str(uuid.uuid4()),
                                          (int(x+w/2),int(y+h)),
                                          pbox,
                                          key_points )
                continue

            prev_points = people['key_points']
            next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None)

            # Filtra solo i punti che sono stati trovati correttamente
            good_prev = prev_points[status == 1]
            good_next = next_points[status == 1]

            # Calcola la media dello spostamento dei punti tracciati
            movement = np.mean(good_next - good_prev, axis=0)

            # Verifica se il bounding box nel secondo frame corrisponde
            x_next, y_next, w_next, h_next = pbox
            next_center = np.array([int(x_next + w_next /2), int(y_next + h_next)])
            x, y, w, h = pbox

            # Se il movimento medio dei punti corrisponde alla posizione stimata del box nel secondo frame, abbiamo un match
            if np.linalg.norm(next_center - (np.array([int(x + w/2), int(y+h)]) + movement)) < 50:

                key_points = cv2.goodFeaturesToTrack(next_gray[y:y + h, x:x + w], maxCorners=100, qualityLevel=0.3,
                                                     minDistance=7,
                                                     blockSize=7)

                new_point = np.array([int(x + w/2), int(y+h)]) + movement
                self.__edit_person_info(people['name'], new_point,  pbox, key_points)

                if self.__track_on_map:
                    self.__trace_point(people['name'], new_point[0], new_point[1])

        if self.__show_people_box:
            frame = self.__show_people(frame, people_boxes)
            frame = self.__draw_map_on_frame(frame)
            frame = self.__draw_map_anchors(frame)

        return frame