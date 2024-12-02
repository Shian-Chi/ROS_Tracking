import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import threading as thrd
import queue
import sys
import signal
from tutorial_interfaces.msg import Bbox  

# Randomly generate RTSP addresses
rtspAddress = 'rtsp://127.0.0.' + str(np.random.randint(0, 256)) + ':8080/video_feed'

frame_queue = queue.Queue(10)
stop_stream = False

def stream(frame_queue):
    global stop_stream
    vidCap = cv2.VideoCapture(rtspAddress)
    while not stop_stream and vidCap.isOpened():
        ret, image = vidCap.read()
        if ret and not frame_queue.full():
            frame_queue.put(image)
    vidCap.release()

# signal processing function
def signal_handler(sig, frame):
    global stop_stream
    stop_stream = True
    print('Shutting down...')
    rclpy.shutdown()
    sys.exit(0)

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("bbox_subscriber")
        self.bboxSub = self.create_subscription(Bbox, "bbox", self.bbox_callback, 10)
        self.detect = False
        self.top_left = []
        self.bottom_right = []

    def bbox_callback(self, msg):
        self.detect = msg.detect
        self.top_left = [msg.x0, msg.y0]
        self.bottom_right = [msg.x1, msg.y1]

class TimerNode(Node):
    def __init__(self, sub: MinimalSubscriber):
        super().__init__("save_node")
        self.sub = sub
        self.timer = self.create_timer(1 / 30, self.process_frame)
        self.bridge = CvBridge()
        self.out = None

    def setup_video_writer(self, frame):
        height, width, _ = frame.shape
        self.out = cv2.VideoWriter('/home/ubuntu/torch_v2/yolo_tracking_v2/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    def process_frame(self):
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Initialize movie storage
            if self.out is None:
                self.setup_video_writer(frame)

            # Draw BBox
            top_left = tuple(self.sub.top_left)
            bottom_right = tuple(self.sub.bottom_right)
            if top_left and bottom_right and self.sub.detect:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                # Write image
                cv2.imwrite('output.jpg', frame)

            # Write video
            self.out.write(frame)

    def destroy_node(self):
        if self.out:
            self.out.release()
        super().destroy_node()

def main(args=None):
    global stop_stream
    stop_stream = False

    # 註冊信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rclpy.init(args=args)

    subscriber = MinimalSubscriber()
    timer_node = TimerNode(subscriber)
    executor = MultiThreadedExecutor()
    executor.add_node(subscriber)
    executor.add_node(timer_node)

    stream_thread = thrd.Thread(target=stream, args=(frame_queue,))
    stream_thread.start()

    try:
        executor.spin()
    finally:
        stop_stream = True
        stream_thread.join()
        timer_node.destroy_node()
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
