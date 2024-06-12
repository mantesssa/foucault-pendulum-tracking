# server.py
import socket
import time
import requests

class TCPServer:
    def __init__(self, tcp_ip='127.0.0.1', tcp_port=9999, buffer_size=4096):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.buffer_size = buffer_size

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.tcp_ip, self.tcp_port))
            server_socket.listen(1)
            print(f"Listening for TCP connections on {self.tcp_ip}:{self.tcp_port}")

            stream_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
            response = requests.get(stream_url, stream=True)

            while response.status_code != 200:
                time.sleep(1)
                response = requests.get(stream_url, stream=True)
                print(f"Failed to connect to stream. Status code: {response.status_code}")

            print("Connected to stream", response.status_code)

            while True:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"Connected to {addr}")
                    try:
                        for chunk in response.iter_content(chunk_size=self.buffer_size):
                            if not chunk:
                                break
                            conn.sendall(chunk)
                            # Simulate delay for testing purposes
                            time.sleep(0.001)
                    except Exception as e:
                        print(f"Error sending stream: {e}")
                    finally:
                        print("Client disconnected.")
                        conn.close()

if __name__ == '__main__':
    server = TCPServer('127.0.0.1', 9999)
    server.start_server()
