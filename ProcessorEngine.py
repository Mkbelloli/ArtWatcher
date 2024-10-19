import uuid
import cv2
import numpy as np
import socket
import utils
import time

PERSON_CLASSID = 0
PERSON_BOX_COLOR = (150,150,0)
PERSON_POINT_COLOR = (250,250, 0)
MAP_LINE_COLOR = (0, 0, 250)
MAP_ANCHOR_COLOR = (0, 0, 250)

IDLE_TIME = 2

class ProcessorEngine:

    def __init__(self, show_reference_lines = True, persone_min_confidence=0.7, track_on_map=True):

        # flag to define behavior
        self.__show_reference_lines = show_reference_lines
        self.__min_confidence_person = persone_min_confidence
        self.__track_on_map = track_on_map

        # YOLOv3 for object identification
        self.__net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # points to design a map
        self.__map_points = []

        # points for homografy
        self.__camera_map_points = []
        self.__real_map_points = []
        self.__H = None

        # sequential frames for time analysis
        self.__curr_frame_gray = None
        self.__prev_frame_gray = None

        # socket to send data
        self.__client_socket = None

        # create the socket to send data
        if self.__track_on_map:
            self.__client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__client_socket.connect(('127.0.0.1', 65432))

        # dictionary of identified people
        self.__people_dict = {}

    def __store_person_info(self, name, map_point, box, key_points):
        """
        store person info in dictionary
        :param name: persona user or id
        :param map_point: point in the map (in pixel)
        :param box: box around the person (x, y, w, h)
        :param key_points: key points related to the person
        :return:
        """

        # log for debugging activities
        msg = f"Add {name}"
        if name in self.__people_dict:
            msg = f"Update {name}"
        utils.print_log(msg)

        if key_points is None and \
            not name in self.__people_dict:
            # new object without keypoint; not saved. A very important info is missing
            return False

        if key_points is None:
            # existing object without updated keypoint; saved with the old value
            key_points = self.__people_dict[name]['key_points']

        self.__people_dict[name]= { 'name': name,
                                    'map_point': map_point,
                                    'box': box,
                                    'key_points': key_points.copy(),
                                    'last_update': time.time()}

        curr_time = time.time()
        people_to_delete = [k for k, v in self.__people_dict.items() if curr_time - v['last_update'] > IDLE_TIME]

        # Cancella le voci dal dizionario originale
        for key in people_to_delete:
            del self.__people_dict[key]

    def __get_people_boxes(self):
        """
        return boxes for each stored person object
        :return: array of boxes (x, y, w, h)
        """
        return [x['box'] for x in  self.__people_dict.values()]

    def __get_people(self, query_box):
        """
        return best people object according input query
        :param query_box: input query to find best object
        :return: best info object or None if not present
        (no object satisfies input query)
        """

        if self.__prev_frame_gray is None or \
            len(self.__people_dict)==0:
            utils.print_log('Initialization without data')
            return -1, None

        # get interest point in frame in bounding box defined by input query box in prev frame
        # prev frame (t-1) is used since the full analysis of movement is performed on the current one
        q_x, q_y, q_w, q_h = query_box
        prev_points = cv2.goodFeaturesToTrack(self.__prev_frame_gray[int(q_y):int(q_y + q_h), int(q_x):int(q_x + q_w)]
                                              , maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if prev_points is not None:
            # coords update according base point (q_x, q_y) in query
            prev_points += np.array([[q_x, q_y]])
        else:
            # no key points, so no best person identified
            return -1, None

        # find next key point in the second frame using Lucas-Kanade OpticalFlow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(self.__prev_frame_gray, self.__curr_frame_gray, prev_points, None)

        # get only valid points
        valid_next_points = next_points[status == 1]

        # count how many valid next points (next position) are contained in people boxes
        counts = []

        people_boxes = self.__get_people_boxes()  # get all boxes for stored people objects

        for pbox in people_boxes:
            x, y, w, h = pbox
            inside_box_count = 0
            for point in valid_next_points:
                # verify if any valid next point is in the current box
                if x <= point[0] <= x + w and y <= point[1] <= y + h:
                    inside_box_count += 1
            counts.append(inside_box_count)

        # find the person object with the highest count
        best_match_index = np.argmax(counts)
        best_match_obj = list(iter(self.__people_dict.values()))[best_match_index]
        best_match_count = counts[best_match_index]

        if best_match_count == 0:
            # there are no boxes that contain at least one key point.
            return -1, None

        # return best choice obj
        return best_match_count, best_match_obj

    def load_mapped_points(self, camera_points, real_points):
        """
        load points that map pixel points and physical points to calculate homografy matrix
        :param camera_points: points from camera (pixel)
        :param real_points: points from map (meter)
        :return:
        """

        # calculate Homography matrix
        self.__real_map_points = np.array(real_points)
        self.__camera_map_points = np.array(camera_points)
        self.__H, status = cv2.findHomography(self.__camera_map_points, self.__real_map_points)

    def load_map(self, points):
        """
        load points to draw maps on the frame
        :param points: map points
        :return:
        """

        self.__map_points = points

    def __draw_map_on_frame(self, frame):
        """
        draw map on the frame
        :param frame: frame to be used
        :return: modified frame
        """

        # used loaded map points to draw the map
        for pi in range(len(self.__map_points)-1, -1, -1):
            cv2.line(frame, self.__map_points[pi], self.__map_points[pi-1], MAP_LINE_COLOR, 1)

        return frame

    def __draw_map_anchors(self, frame):
        """
        draw map anchors on the frame
        :param frame: frame to be used
        :return: modified frame
        """

        # used loaded map points to draw the anchors
        for anch_p in self.__map_points:
            cv2.circle(frame, center=anch_p, radius=5, color=MAP_ANCHOR_COLOR, thickness=cv2.FILLED)
        return frame

    def __show_people(self, frame, people_boxes):
        """
        draw boxes for each identified person in the frame
        :param frame: frame to be used
        :param people_boxes: boxes to be drawn
        :return: modified frame
        """

        # used boxes for stored info object
        for box in people_boxes:
            color = PERSON_BOX_COLOR
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), color, 1)
            cv2.circle(frame, (int(box[0]+box[2]/2), int(box[1]+box[3])), 5, PERSON_POINT_COLOR, cv2.FILLED)

        return frame

    def __trace_point(self, name, x, y):
        """
        sent new coordinates to the UI server
        :param name: person name or id to be sent
        :param x: x coord of the person
        :param y: y coord of the person
        :return:
        """

        # transform coords using homography matrix
        transformed_point = cv2.perspectiveTransform(np.array([[[x, y]]]), self.__H)

        # local temporary variables for a easier analysis
        x_map = transformed_point[0][0][0]
        y_map = transformed_point[0][0][1]

        # send updating coordinates for <name> person object
        message = f"{name} {x_map} {y_map} <END>"
        utils.print_log(f"Sent: {message}")
        self.__client_socket.send(message.encode('utf-8'))

    def __get_people_from_frame(self, frame):
        """
        get people in a frame
        :param frame: frame to be used
        :return: list of identified boxes
        """

        # YOLO object identification
        H, W = frame.shape[:2]
        ln = self.__net.getLayerNames()
        ln = [ln[i - 1] for i in self.__net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.__net.setInput(blob)
        outputs = self.__net.forward(ln)
        outputs = np.vstack(outputs)

        boxes = []           # store boxes
        confidences = []     # confidence per box
        map_polygon = np.array(self.__map_points, dtype=np.int32) # map polygon

        for output in outputs:

            confidence_array = output[5:]
            classID = np.argmax(confidence_array)

            x, y, w, h = output[:4] * np.array([W, H, W, H])
            # x e y are related to center of the box
            # since box reference is related to the upper left corner
            # so it needs to change coords
            x = int(x-w/2)
            y = int(y-h/2)

            # check of the base point of the box
            if classID != PERSON_CLASSID or \
               confidence_array[classID] < self.__min_confidence_person or \
               cv2.pointPolygonTest(map_polygon, (int(x + (w / 2)), int(y + h)), False) < 0:
                # continue if:
                #  - not a person
                #  - low confidence in identification
                #  - base point outside the map
                continue

            # append box and confidence for the filtered box
            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(confidence_array[classID])

        # filter and reduce the number of overlapping bounding boxes generated.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.__min_confidence_person, self.__min_confidence_person - 0.1)

        # get people boxes
        people_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                # x,y,w,h
                people_boxes.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))

        return people_boxes

    def process_frame(self, frame):
        """
        full process of the frame
        :param frame: frame to be analyzed
        :return:
        """

        # save in internal variables current and previous frame
        self.__prev_frame_gray = self.__curr_frame_gray.copy() if not self.__curr_frame_gray is None else None
        self.__curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get boxes from current frame
        people_boxes = self.__get_people_from_frame(frame)

        if self.__prev_frame_gray is None:
            # for the first frame (prev_frame is None) continue
            return frame

        # list on identified boxes
        for pbox in people_boxes:

            # get person object for current box
            score, people = self.__get_people(pbox)

            if people is None:
                # no object found

                x, y, w, h = pbox

                # create key points
                key_points = cv2.goodFeaturesToTrack(self.__curr_frame_gray[int(y):int(y + h), int(x):int(x + w)], maxCorners=100, qualityLevel=0.2,
                                                      minDistance=7,
                                                      blockSize=7)
                if not key_points is None:
                    key_points += np.array([[int(x), int(y)]])

                # add new object
                name = str(uuid.uuid4())
                self.__store_person_info( name,
                                          (int(x+w/2),int(y+h)),
                                          pbox,
                                          key_points )
                if self.__track_on_map:
                    # send position to UI server
                    self.__trace_point(name, x+w/2, y+h)
                continue

            prev_points = people['key_points']

            next_points, status, err = cv2.calcOpticalFlowPyrLK(self.__prev_frame_gray, self.__curr_frame_gray, prev_points, None)

            # Filter key points correctly identified
            good_prev = prev_points[status == 1]
            good_next = next_points[status == 1]

            # movement identification
            movement = np.mean(good_next - good_prev, axis=0)

            x_next, y_next, w_next, h_next = pbox
            next_center = np.array([int(x_next + w_next /2), int(y_next + h_next)])
            x, y, w, h =  people['box']

            # If the average movement matches the estimated position in the second frame, we have a match.
            if np.linalg.norm(next_center - (np.array([int(x + w/2), int(y+h)]) + movement)) < 50:

                # get key points
                key_points = cv2.goodFeaturesToTrack(self.__curr_frame_gray[y_next:y_next + h_next, x_next:x_next + w_next], maxCorners=100, qualityLevel=0.2,
                                                     minDistance=3, blockSize=7)

                new_point = np.array([int(x + w/2), int(y+h)]) + movement

                # update position of object already stored
                self.__store_person_info(people['name'], new_point, pbox, key_points)

                if self.__track_on_map:
                    # send position to UI server
                    self.__trace_point(people['name'], new_point[0], new_point[1])

        if self.__show_reference_lines:

            # show box around people
            frame = self.__show_people(frame, people_boxes)

            # show map
            frame = self.__draw_map_on_frame(frame)

            # show map anchors
            frame = self.__draw_map_anchors(frame)

        return frame