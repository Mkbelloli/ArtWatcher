import pyglet
import socket
import threading
import random
from pyglet import shapes
class PeopleTrackingUI:
    def __init__(self, width, height):
        self.window = pyglet.window.Window( width, height)
        self.batch = pyglet.graphics.Batch()
        self.__width = width
        self.__height = height
        self.__x_margin = 10
        self.__y_margin = 10
        self.__base = shapes.Rectangle(self.__x_margin, self.__y_margin, width-2*self.__x_margin, height--2*self.__y_margin, color=(30, 75, 70), batch=self.batch)
        self.window.push_handlers(self.on_draw)
        self.__people = {}

    def add_person(self, userid, x, y):
        x += self.__x_margin
        y += self.__y_margin
        symb = shapes.Circle(x, y, 10, color=(50, 150, 255), batch=self.batch)
        self.__people[userid] = (x, y, symb)

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

                        userid, str_x, str_y = data.split(sep=' ')

                        # Creare un oggetto nel thread principale di Pyglet
                        pyglet.clock.schedule_once(lambda dt: self.add_person(userid, int(str_x), int(str_y)), 0)
                        conn.sendall(f"Oggetto creato per comando: {data}".encode('utf-8'))


    def run(self):

        # Avvia il server socket su un thread separato
        socket_thread = threading.Thread(target=self.run_socket_server)
        socket_thread.daemon = True  # Il thread terminerà automaticamente quando il programma si chiude
        socket_thread.start()

        # Avvia Pyglet nel thread principale
        self.app_run()


if __name__ == "__main__":
    ui = PeopleTrackingUI(640, 480)
    ui.run()
