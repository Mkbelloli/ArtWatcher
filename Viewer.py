import pyglet
import socket
import threading
import random
import time
from pyglet import shapes
import utils

PIXEL_PER_METER = 40
X_MARGIN = 10
Y_MARGIN = 10
MAX_IDLE_TIME = 3

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 65432

class PeopleTrackingUI:
    """
    UI to track people identified in video.
    It is composed by two parts that works two different threads:
        - Server to receive person positions
        - Graphical UI to draw people's position
    """


    def __init__(self, width, height):

        # size window with with margins
        self.__width = width * PIXEL_PER_METER + 2 * X_MARGIN
        self.__height = height * PIXEL_PER_METER + 2 * Y_MARGIN

        # window components
        self.window = pyglet.window.Window( self.__width, self.__height)
        self.batch = pyglet.graphics.Batch()

        # draw map
        self.__base = shapes.Rectangle(X_MARGIN, Y_MARGIN, self.__width-2*X_MARGIN, self.__height-2*Y_MARGIN, color=(30, 75, 70), batch=self.batch)

        # draw handler
        self.window.push_handlers(self.on_draw)

        # people set
        self.__people = {}

    def update_person(self, userid, x, y):
        """
        Update position information for a specific person
        :param userid: person name or id
        :param x: X position in pixel
        :param y: Y position in pixel
        :return: None
        """

        # added margins to position
        x += X_MARGIN
        y += Y_MARGIN

        # drawing object creation (according pyglet)
        symb = shapes.Circle(x, y, 10, color=(50, 150, 255), batch=self.batch)

        # saved person info (name, position and last udpdating time)
        self.__people[userid] = (x, y, symb, time.time())

        # remove any person not updated for more than MAX_IDLE_TIME seconds.
        self.__remove_old_people()

    def __remove_old_people(self):
        """
        Remove any person not updated for more than MAX_IDLE_TIME seconds.
        :return:
        """

        ct = time.time()
        self.__people = {key: value for key, value in self.__people.items() if (ct-value[3])< MAX_IDLE_TIME}

    def app_run(self):
        """
        app method to run pyglet
        :return:
        """

        pyglet.app.run()

    def on_draw(self):
        """
        Rendering function
        :return:
        """

        self.window.clear()
        self.batch.draw()

    def run_socket_server(self):
        """
        main function to run server
        :return:
        """

        # created socket server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:

            # binding
            server_socket.bind((SERVER_HOST, SERVER_PORT))
            server_socket.listen()

            utils.print_log(f"Server in ascolto su {SERVER_HOST}:{SERVER_PORT}...")

            while True:
                conn, addr = server_socket.accept()
                with conn:

                    utils.print_log(f"Connesso da {addr}")

                    # reading received data
                    while True:

                        data = conn.recv(1024).decode('utf-8')
                        if not data:
                            break

                        utils.print_log(f"Received: {data}")

                        # splitting received data per <END> tag
                        while '<END>' in data:

                            message, data = data.split('<END>', 1)
                            message = message.strip()
                            data = data.strip()
                            userid, str_x, str_y = message.split(sep=' ')

                            # mapping meter x pixel4meter = pixels
                            map_x = int(float(str_x) * PIXEL_PER_METER )
                            map_y = int(float(str_y) * PIXEL_PER_METER )

                            # Update position for userid
                            self.update_person(userid, map_x, map_y)

                            # ack message, for debug only
                            conn.sendall(f"Oggetto creato per comando: {message}".encode('utf-8'))


    def run(self):
        """
        run method to start threads
        :return:
        """

        # run server socket
        socket_thread = threading.Thread(target=self.run_socket_server)
        socket_thread.daemon = True  # Il thread terminerà automaticamente quando il programma si chiude
        socket_thread.start()

        # run UI
        self.app_run()


if __name__ == "__main__":
    utils.deactivate_logs()
    ui = PeopleTrackingUI(16, 8)
    ui.run()
