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
        self.cascade = cv2.CascadeClassifier('./weights/cascade.xml')


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

    def disconnect(self):
        print("disconnecting")  # do not work
        if self.client_socket:
            self.client_socket.close()

    def detect_ball_haar(self, image, prev_detection):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=35, minSize=(14, 14), maxSize=(32, 32))
        # print("detections", detections)

        nearest_detect = None
        if len(detections) > 0:
            if prev_detection is None:
                x, y, w, h = detections[0]
                return x, y, x+w, y+h

            min_dist = float('inf')
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_detection
            prev_cx, prev_cy = (prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2

            for detect in detections:
                x, y, w, h = detect
                detect = (x, y, x+w, y+h)

                now_x1, now_y1, now_x2, now_y2 = detect
                now_cx, now_cy = (now_x1 + now_x2) / 2, (now_y1 + now_y2) / 2
                dist = np.sqrt((now_cx - prev_cx) ** 2 + (now_cy - prev_cy) ** 2)

                if dist < min_dist:
                    min_dist = dist
                    nearest_detect = detect

        return nearest_detect




    def draw_vector(self, image, detection, prev_position):
        x1, y1, x2, y2 = detection
        px1, py1, px2, py2 = prev_position

        vx = px1 - x1
        vy = py1 - y1
        x_c = int((x1 + x2)/2)
        y_c = int((y1 + y2)/2)

        cv2.arrowedLine(image, (x_c, y_c), (int(x_c - vx * 1.5), int(y_c - vy * 1.5)), (255, 0, 0), 2)

    def detect(self):
        video_stream = self.get_video_stream()
        frame_id = 0
        roi = (210, 120, 400, 300)  # Region of interest coordinates
        crop_x1, crop_y1, crop_x2, crop_y2 = roi
        prev_detection = None

        for frame in video_stream:
            now_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            detection = self.detect_ball_haar(now_frame, prev_detection)

            if detection is not None:
                x1, y1, x2, y2 = detection
                x1 = crop_x1 + x1
                y1 = crop_y1 + y1

                x2 = crop_x1 + x2
                y2 = crop_y1 + y2
                detection = (x1, y1, x2, y2)

                cv2.rectangle(frame, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), 2)
                if prev_detection is not None:
                    self.draw_vector(frame, detection, prev_detection)

                add_detection_data(frame_id, [detection])

            cv2.imshow("Ball Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev_detection = detection
            frame_id += 1

        self.disconnect()
        cv2.destroyAllWindows()

def run_server():
    server = CircleDetectionServer('localhost:50052')
    server.start()

if __name__ == '__main__':
    streaming_server_thread = threading.Thread(target=run_server) # в паралельном потоке запускаем grpc сервер для отпавки координат
    streaming_server_thread.start()

    client = TCPClient()
    client.connect()
    client.detect()
