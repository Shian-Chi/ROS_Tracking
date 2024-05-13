import math
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import NavSatFix, Imu
from mavros_msgs.msg import Altitude
from transforms3d import euler

from tutorial_interfaces.msg import Img, Bbox
# from mavros_msgs.msg import  Bbox, Img

import sys
import os

# 獲得當前文件的絕對路徑，然後找到上層目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一層目錄

# 將這個目錄添加到 sys.path
sys.path.insert(0, parent_dir)
from trackDetect import pubImgData, pubImgData

img_data = pubImgData
bbox_data = pubImgData

def radian_conv_degree(Radian):
    return ((Radian / math.pi) * 180)

rclpy.init()
class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        self.imgPublish = self.create_publisher(Img, "img", 10)
        img_timer_period = 1/35
        self.img_timer = self.create_timer(img_timer_period, self.img_callback)

        self.bboxPublish = self.create_publisher(Bbox, "bbox", 10)
        bbox_timer_period = 1/10
        self.img_timer = self.create_timer(bbox_timer_period, self.bbox_callback)

        self.img = Img()
        self.bbox = Bbox()

    def img_callback(self):
        self.img.first_detect, self.img.second_detect, self.img.third_detect, self.img.camera_center, self.img.motor_pitch, \
            self.img.motor_yaw, self.img.target_latitude, self.img.target_longitude, self.img.hold_status, self.img.send_info = img_data.values()

        self.imgPublish.publish(self.img)

    def bbox_callback(self):
        self.bbox.x0 = int(bbox_data['x0'])
        self.bbox.y0 = int(bbox_data['y0'])
        self.bbox.x1 = int(bbox_data['x1'])
        self.bbox.y1 = int(bbox_data['y1'])
        self.bboxPublish.publish(self.bbox)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        # self.subscription = self.create_subscription(Img,"topic",self.holdcb,10)
        self.GlobalPositionSuub = self.create_subscription(NavSatFix, "mavros/global_position/global", self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, "mavros/imu/data", self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, "img", self.holdcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.hold = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0

    def holdcb(self, msg):
        self.hold = img_data["hold_status"] = msg.hold_status

    def GPcb(self, msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.gps_altitude = msg.altitude

    def IMUcb(self, msg: Imu):
        ned_euler_data = euler.quat2euler([msg.orientation.w,
                                           msg.orientation.x,
                                           msg.orientation.y,
                                           msg.orientation.z])
        self.pithch = radian_conv_degree(ned_euler_data[0])
        self.roll = radian_conv_degree(ned_euler_data[1])
        self.yaw = radian_conv_degree(ned_euler_data[2])

    def getImuPitch(self):
        return self.pitch

    def getImuYaw(self):
        return self.yaw

    def getImuRoll(self):
        return self.roll

    def getHold(self):
        return img_data["hold_status"]

    def getLatitude(self):
        return self.latitude

    def getLongitude(self):
        return self.longitude

    def getAltitude(self):
        return self.gps_altitude


pub = MinimalPublisher()
sub = MinimalSubscriber()


def _spinThread(publish=pub, subscribe=sub):
    executor = MultiThreadedExecutor()
    executor.add_node(publish)
    executor.add_node(subscribe)
    executor.spin()



