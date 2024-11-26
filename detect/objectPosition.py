from detect_function import YoloDetector, YOLO_parameter
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functools import partial

from ctrl.gimbal_PID import GimbalTimerTask, yaw, pitch
from ctrl.pid.parameter import Parameters

import threading as thrd
import signal
import queue, math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import NavSatFix, Imu
from transforms3d import euler
from tutorial_interfaces.msg import Img, Bbox, Lidar, MotorInfo
# from mavros_msgs.msg import Altitude, Lidar, Bbox, Img


pub_img = {"detect": False,
           "camera_center": False,
           "motor_pitch": 0.0,
           "motor_yaw": 0.0,
           "target_latitude": 0.0,
           "target_longitude": 0.0,
           "hold_status": False,
           "send_info": False
           }


pub_bbox = {
    'detect' : False,
    'class_id': 0,
    'confidence': 0.0,
    'x0': 1280,
    'x1': 0,
    'y0': 720,
    'y1': 0
}


para = Parameters()


def signal_handler(sig, frame):
    global yaw, pitch, executor
    print('Signal detected, shutting down gracefully')
    yaw.stop()
    pitch.stop()
    executor.shutdown()
    rclpy.shutdown()
    sys.exit(0)
    

def radian_conv_degree(Radian: float) -> float:
    return ((Radian / math.pi) * 180)


rclpy.init(args=None)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.GlobalPositionSuub = self.create_subscription(NavSatFix, "mavros/global_position/global", self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, "mavros/imu/data", self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, "img", self.holdcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # self.gimbalRemove = self.create_subscription(GimbalDegree, "gimDeg", self.gimAngDegcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.distance = self.create_subscription(Lidar, "lidar", self.lidarcb, 1)
        self.bboxPredcd = self.create_subscription(Bbox, 'bbox', self.bboxcb, 1)
        self.motorcb = self.create_subscription(MotorInfo, 'motor_info', self.motorInfocb, 1)
        
        self.hold = False
        
        # GPS
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        
        # Drone attitude
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.drone_yaw = 0.0
        
        # Gimbal attitude
        self.gimbalYaw = 0.0
        self.gimbalPitch = 0.0
        
        # Lidar Measure distance
        self.discm = 0.0
        
        # YOLO Bbox
        self.detect = False
        self.ID = -1
        self.conf = -1
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        
        self.gimbalYawDeg = 0.0
        self.gimbalPitchDeg = 0.0

    def gimAngDegcb(self, msg):
        self.gimbalYaw = msg.yaw
        self.gimbalPitch = msg.pitch
    
    def holdcb(self, msg):
        self.hold = pub_img["hold_status"] = msg.hold_status

    def GPcb(self, msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.gps_altitude = msg.altitude

    def IMUcb(self, msg):
        ned_euler_data = euler.quat2euler([msg.orientation.w,
                                           msg.orientation.x,
                                           msg.orientation.y,
                                           msg.orientation.z])
        self.drone_pithch = radian_conv_degree(ned_euler_data[0])
        self.drone_roll = radian_conv_degree(ned_euler_data[1])
        self.drone_yaw = radian_conv_degree(ned_euler_data[2])

    def lidarcb(self, msg):
        self.discm = msg.distance_cm

    def bboxcb(self, msg):
        self.detect = msg.detect
        self.ID = msg.class_id
        self.conf = msg.confidence
        self.x0 = msg.x0
        self.y0 = msg.y0
        self.x1 = msg.x1
        self.y1 = msg.y1
    
    def motorInfocb(self, msg):
        self.gimbalYawDeg = pub_img['motor_yaw'] = msg.yaw_angle
        self.gimbalPitchDeg = pub_img['motor_pitch'] = msg.pitch_angle
    
    def get_bbox(self):
        # print(f"get_bbox: {self.detect}")
        return self.x0, self.y0, self.x1, self.y1

ROS_Sub = MinimalSubscriber()
gimbalTask = GimbalTimerTask(ROS_Sub)


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        # Img publish
        self.imgPublish = self.create_publisher(Img, "img", 0)
        img_timer_period = 1/25
        self.img_timer = self.create_timer(img_timer_period, self.img_callback)
        
        # Bbox publish
        self.bboxPublish = self.create_publisher(Bbox, "bbox", 0)
        bbox_timer_period = 1/25
        self.img_timer = self.create_timer(bbox_timer_period, self.bbox_callback)
        
        self.img = Img()
        self.bbox = Bbox()
        
    def img_callback(self):
        pub_img['camera_center'] = gimbalTask.bbox_center
        pub_img['motor_pitch'] = pub_img['motor_pitch'] + ROS_Sub.drone_pitch
        pub_img['motor_yaw'] = pub_img['motor_yaw']
        self.img.detect, self.img.camera_center, self.img.motor_pitch, self.img.motor_yaw, \
            self.img.target_latitude, self.img.target_longitude, self.img.hold_status, self.img.send_info = pub_img.values()        
        self.imgPublish.publish(self.img)
    
    def bbox_callback(self):
        self.bbox.detect = pub_bbox['detect']
        self.bbox.class_id = pub_bbox['class_id']
        self.bbox.confidence = pub_bbox['confidence']

        self.bbox.x0 = pub_bbox['x0']
        self.bbox.y0 = pub_bbox['y0']

        self.bbox.x1 = pub_bbox['x1']
        self.bbox.y1 = pub_bbox['y1']

        # Publish BoundingBox message
        self.bboxPublish.publish(self.bbox)
    
ROS_Pub = MinimalPublisher()


def _spinThread(*args):
    global executor
    executor = MultiThreadedExecutor()

    for task in args:
        executor.add_node(task)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        rclpy.shutdown()
        

def Update_pub_bbox(detect=False, id=0, conf=0.0, x0=0, y0=0, x1=0, y1=0):
    # updata pub_bbox
    pub_bbox['detect'] = detect
    pub_bbox['class_id'] = int(id)
    pub_bbox['confidence'] = float(conf)
    pub_bbox['x0'] = int(x0)
    pub_bbox['x1'] = int(x1)
    pub_bbox['y0'] = int(y0)
    pub_bbox['y1'] = int(y1)
              

def bbox_filter(xyxy0, xyxy1):
    c0 = [((xyxy0[0] + xyxy0[2]) / 2), ((xyxy0[1] + xyxy0[3]) / 2)]
    c1 = [((xyxy1[0] + xyxy1[2]) / 2), ((xyxy1[1] + xyxy1[3]) / 2)]
    
    dis = math.sqrt(((c1[0] - c0[0])**2) + ((c1[1] - c0[1])**2))
    return dis<=256, dis
        

def detection_hold_count():
    count = 0
    status = False

    def inner_detection(stat):
        nonlocal count, status  # Use outer variables
        if stat:
            count += 1
        else:
            count = 0
        status = count >= 4
        return status

    wrapped_function = partial(inner_detection)
    wrapped_function.count = lambda: count
    wrapped_function.status = lambda: status 
    return wrapped_function

isContinuous = detection_hold_count()

    
            
def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ROS
    global ROS_Pub, ROS_Sub

    ROS_spin = thrd.Thread(target=_spinThread, args=(ROS_Pub, ROS_Sub, gimbalTask))
    ROS_spin.start()
    
    # YOLO
    yoloPara = YOLO_parameter
    detector = YoloDetector(*yoloPara)

if __name__ == '__main__':
    main()
