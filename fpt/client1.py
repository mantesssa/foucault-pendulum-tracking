import cv2
import numpy as np
import socket
from ultralytics import YOLO
import threading

import grpc
from concurrent import futures
import time
import queue
import circle_detection_pb2
import circle_detection_pb2_grpc

# Глобальная очередь для хранения данных
data_queue = queue.Queue()

class CircleDetectionService(circle_detection_pb2_grpc.CircleDetectionServiceServicer):
    def GetStreamData(self, request, context):
        while True:
            try:
                data = data_queue.get(timeout=10)  # Ожидаем данные в течение 10 секунд
                yield data
            except queue.Empty:
                return

class CircleDetectionServer:
    def __init__(self, server_address):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.service = CircleDetectionService()
        circle_detection_pb2_grpc.add_CircleDetectionServiceServicer_to_server(self.service, self.server)
        self.server.add_insecure_port(server_address)
        self.server_address = server_address

    def start(self):
        self.server.start()
        print(f"Server started. Listening on {self.server_address}...")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)

    def stop(self):
        print("Stopping server...")
        self.server.stop(0)
        print("Server stopped.")


def add_detection_data(frame_id, detections):
    data = circle_detection_pb2.Data(
        frame_id=frame_id,
        detections=[
            circle_detection_pb2.Detection(
                object_id=1,
                bbox=circle_detection_pb2.BoundingBox(
                    x_min=detection[0],
                    y_min=detection[1],
                    x_max=detection[2],
                    y_max=detection[3]
                )
            ) for detection in detections
        ]
    )
    data_queue.put(data)




class TCPClient:
    def __init__(self, server_ip='127.0.0.1', server_port=9999, grpc_port=50052):
        self.server_ip = server_ip
        self.server_port = server_port
        self.grpc_port = grpc_port
        self.client_socket = None
        self.model = YOLO('./weights/best.pt')


    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        print("Connected to server")


    def get_video_stream(self):
        bytes = b''
        while True:
            chunk = self.client_socket.recv(1024)
            if not chunk:
                print("no data recved")
                break
            bytes += chunk
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                yield frame

    # def disconnect(self):
    #     print("disconnecting")  # do not work
    #     if self.client_socket:
    #         print("disconecting???")
    #         self.client_socket.close() # виснет здесь

    def disconnect(self):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            print("Disconnected from server")


    def get_nearest_circle(self, circles, previous_circle):
        # cv2.circle(frame, (x1 + 100, y1 + 90), 12, (0, 255, 0), 4)
        if previous_circle is None:
            return circles[0]  # Return the first detecton if there is no previous detection

        min_dist = float('inf')
        nearest_circle = None
        prev_x, prev_y, _ = previous_circle

        for circle in circles:
            x, y, r = circle
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_circle = circle
        return nearest_circle



    def draw_ball_and_vector(self, image, detection, prev_position):
        x1, y1, x2, y2 = detection
        px1, py1, px2, py2 = prev_position

        vx = px1 - x1
        vy = py1 - y1
        x_c = int((x1 + x2)/2)
        y_c = int((y1 + y2)/2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.arrowedLine(image, (x_c, y_c), (int(x_c - vx * 1.5), int(y_c - vy * 1.5)), (255, 0, 0), 2)

    def detect(self):
        video_stream = self.get_video_stream()

        roi = (210, 120, 400, 300)  # Region of interest coordinates
        crop_x1, crop_y1, crop_x2, crop_y2 = roi
        # kernel = np.ones((3, 3), np.uint8)



        prev_position = None
        frame_id = 0
        previous_circle = None
        detection = None
        object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=True)



        for frame in video_stream:
            now_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            gray = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)
            mask = object_detector.apply(gray)

            gray_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
            # cv2.imshow('Foucault Pendulum Detection', gray_blurred)

            circles = cv2.HoughCircles(
                gray_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=10,
                param1=40,
                param2=20,
                minRadius=7,
                maxRadius=16
            )

            # If circles are detected, draw them on the original frame
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                nearest_circle = self.get_nearest_circle(circles, previous_circle)
                if nearest_circle is not None:
                    x, y, r = nearest_circle
                    x1 = crop_x1 + x - r
                    y1 = crop_y1 + y - r
                    x2 = crop_x1 + x + r
                    y2 = crop_y1 + y + r
                    detection = (x1, y1, x2, y2)

                    if previous_circle is not None:
                        self.draw_ball_and_vector(frame, detection, prev_position)

                    add_detection_data(frame_id, [detection])

                    previous_circle = nearest_circle
                    # cv2.circle(frame, (x1 + x, y1 + y), r, (0, 255, 0), 4)
                else:
                    previous_circle = None

            # Display the frame with detected circles
            cv2.imshow('Foucault Pendulum Detection', frame)
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev_position = detection
            frame_id += 1

        self.disconnect()
        cv2.destroyAllWindows()


def run_server():
    server = CircleDetectionServer('localhost:50052')
    server.start()

if __name__ == '__main__':
    streaming_server_thread = threading.Thread(target=run_server) # в паралельном потоке запускаем grpc сервер для отпавки координат
    streaming_server_thread.start()

    client = TCPClient('127.0.0.1', 9999)
    client.connect()
    client.detect()
