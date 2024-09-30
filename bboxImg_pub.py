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

# 使用隨機生成的 rtspAddress
rtspAddress = 'rtsp://10.242.36.97:8080/test'

# 設置全局影像框和 bbox 參數
frame = queue.Queue(10)
bbox_coords = {'top_left': [0, 0], 'bottom_right': [0, 0]}

bboxTop = [0, 0]
bboxBottom = [0, 0]

def stream(cap, frame):
    # 開啟 RTSP 串流
    
    while True:
        # 從 RTSP 串流讀取一張影像
        ret, image = cap.read()
        if ret:
            if not frame.full():
                frame.put(image)
                # print("Image added to queue")  # Debug 信息
        else:
            # 若沒有影像跳出迴圈
            break

    # 釋放資源
    cap.release()

# 信號處理函數，用於優雅地關閉節點
def signal_handler(sig, frame):
    global ROS_Pub, ROS_Sub
    print('Signal detected, shutting down gracefully...')

    if ROS_Pub is not None:
        ROS_Pub.destroy_node()
    if ROS_Sub is not None:
        ROS_Sub.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("bboxImg_publisher")
        self.ros_pub_image = None
        self.publisher_ = self.create_publisher(Image, "image", 10)  
        t = 1/60
        self.timer = self.create_timer(t, self.bboxImg_callback)  # 每0.1秒發布一次圖像
        self.bridge = CvBridge()  # OpenCV圖像與ROS圖像之間的轉換
        
    def bboxImg_callback(self):
        if frame.full():
            # 從 queue 中取出影像
            image = frame.get()

            # 獲取當前 bbox 座標，並檢查座標是否有效
            try:
                ret, id, conf, point = ROS_Sub.getBbox()
                print(f"bbox: {point}")
                print(f"Current bbox_coords: {point}")  # 調試輸出座標
               
                # 在影像上畫出矩形框
                if ret:
                    cv2.rectangle(image, (point[0],point[1]), (point[2],point[3]), (0, 255, 0), 5)  # 綠色框

            except Exception as e:
                self.get_logger().error(f"Error drawing bbox: {e}")
                return

            # 将OpenCV图像转换为ROS图像消息
            self.ros_pub_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.publisher_.publish(self.ros_pub_image)
            # self.get_logger().info('Publishing image with bbox')
            if self.ros_pub_image is not None:
                # image = cv2.GaussianBlur(image, (5, 5), 0)
                cv2.imshow("bbox_images", image)
                cv2.waitKey(1)  # 1 millisecond


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("bbox_subscriber")
        # 訂閱 bbox topic，並設置回調函數
        self.bboxSub = self.create_subscription(Bbox, "bbox", self.bboxcb, 5)
        self.detectSTAT = False
        self.ID = 0
        self.conf = 0.0
        self.bboxTop = [0, 0]
        self.bboxBottom = [0, 0]
        
    def bboxcb(self, msg):
        # 更新全局 bbox 座標，將從 topic 接收到的座標存儲到 bbox_coords 中
        self.detectSTAT = msg.detect
        self.bboxTop = [msg.x0, msg.y0]
        self.bboxBottom = [msg.x1, msg.y1]
        # self.get_logger().info(f"Received bbox with class_id: {msg.class_id}, confidence: {msg.confidence}")

    def getBbox(self):
        return self.detectSTAT, self.ID, self.conf, self.bboxTop + self.bboxBottom
        
def main(args=None):
    # 信號處理器設置，捕獲 SIGINT 和 SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rclpy.init(args=args)

    # 初始化 MinimalSubscriber 和 MinimalPublisher 節點
    global ROS_Sub, ROS_Pub
    ROS_Sub = MinimalSubscriber()
    ROS_Pub = MinimalPublisher()

    # 使用 MultiThreadedExecutor 同時處理多個節點
    executor = MultiThreadedExecutor()
    executor.add_node(ROS_Sub)
    executor.add_node(ROS_Pub)

    # 初始化 RTSP 串流讀取的執行緒
    global vidCap
    vidCap = cv2.VideoCapture(rtspAddress)
    streamthrd = thrd.Thread(target=stream, args=(vidCap,frame)) 
    streamthrd.start()

    # 使用 executor 同時處理 MinimalPublisher 和 MinimalSubscriber
    try:
        executor.spin()
    finally:
        ROS_Sub.destroy_node()
        ROS_Pub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
