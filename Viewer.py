import pyglet
import socket
import threading
import random
import time
from pyglet import shapes

PIXEL_PER_METER = 40
X_MARGIN = 10
Y_MARGIN = 10
MAX_IDLE_TIME = 3

class PeopleTrackingUI:
    def __init__(self, width, height):
        self.__width = width * PIXEL_PER_METER + 2 * X_MARGIN
        self.__height = height * PIXEL_PER_METER + 2 * Y_MARGIN
        self.window = pyglet.window.Window( self.__width, self.__height)
        self.batch = pyglet.graphics.Batch()
        self.__base = shapes.Rectangle(X_MARGIN, Y_MARGIN, self.__width-2*X_MARGIN, self.__height-2*Y_MARGIN, color=(30, 75, 70), batch=self.batch)

        self.window.push_handlers(self.on_draw)
        self.__people = {}

    def add_person(self, userid, x, y):
        x += X_MARGIN
        y += Y_MARGIN
        symb = shapes.Circle(x, y, 10, color=(50, 150, 255), batch=self.batch)
        
        self.__people[userid] = (x, y, symb, time.time())
        self.__remove_old_points()

    def __remove_old_points(self):
        ct = time.time()
        self.__people = {key: value for key, value in self.__people.items() if (ct-value[3])< MAX_IDLE_TIME}

    def edit_person(self, userid, x, y):
        self.__people[userid].x = x
        self.__people[userid].y = y

    def remove_person(self, userid):
        del self.__people[userid]

    def app_run(self):
        pyglet.app.run()

    def on_draw(self):
        """Funzione di rendering."""
        self.window.clear()
        self.batch.draw()


    def run_socket_server(self):
        HOST = '127.0.0.1'
        PORT = 65432

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, PORT))
            server_socket.listen()
            print(f"Server in ascolto su {HOST}:{PORT}...")

            while True:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"Connesso da {addr}")
                    while True:
                        data = conn.recv(1024).decode('utf-8')
                        if not data:
                            break
                        #print(f"Received: {data}")
                        while '<END>' in data:
                            message, data = data.split('<END>', 1)
                            message = message.strip()
                            data = data.strip()
                            userid, str_x, str_y = message.split(sep=' ')

                            map_x = int(float(str_x) * PIXEL_PER_METER )
                            map_y = int(float(str_y) * PIXEL_PER_METER )

                            # Creare un oggetto nel thread principale di Pyglet
                            self.add_person(userid, map_x, map_y)
                            #pyglet.clock.schedule_once(lambda dt: self.add_person(userid, map_x, map_y), 0)
                            conn.sendall(f"Oggetto creato per comando: {message}".encode('utf-8'))

    def run(self):

        # Avvia il server socket su un thread separato
        socket_thread = threading.Thread(target=self.run_socket_server)
        socket_thread.daemon = True  # Il thread terminerà automaticamente quando il programma si chiude
        socket_thread.start()

        # Avvia Pyglet nel thread principale
        self.app_run()


if __name__ == "__main__":
    ui = PeopleTrackingUI(16, 16)
    ui.run()
