import grpc
import circle_detection_pb2
import circle_detection_pb2_grpc

def get_data():
    with grpc.insecure_channel('localhost:50052') as channel:
        stub = circle_detection_pb2_grpc.CircleDetectionServiceStub(channel)
        response_iterator = stub.GetStreamData(circle_detection_pb2.RequestData(message=""))

        for data in response_iterator:
            # вывод полученных данных
            print("Received data: ", data)


if __name__ == '__main__':
    get_data()
