import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import NavSatFix, Imu
from mavros_msgs.msg import Altitude
from transforms3d import euler

import time
import math
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

from pid.pid import PID_Ctrl
from pid.parameter import Parameters
from pid.motor import motorCtrl
from pid.position import verticalTargetPositioning
import threading
import queue
from tutorial_interfaces.msg import Img, Bbox, GimbalDegree
#from mavros_msgs.msg import  Bbox, Img 

pub_img = {"detect": False,
           "camera_center": False,
           "motor_pitch": 0.0,
           "motor_yaw": 0.0,
           "target_latitude": 0.0,
           "target_longitude": 0.0,
           "hold_status": False,
           "send_info": False
           }

pub_bbox = {'x0': 0,
        'x1': 0,
        'y0': 0,
        'y1': 0
        }

rclpy.init()

para = Parameters()
pid = PID_Ctrl()
position = verticalTargetPositioning()


def delay(s: int):
    s += 1
    for i in range(1, s):
        print(f"{i}s")
        time.sleep(1)


def print_detection_info(s, detect_count, end_inference, start_inference, end_nms):
    inferenceTime = 1E3 * (end_inference - start_inference)
    NMS_Time = 1E3 * (end_nms - end_inference)
    total_time = inferenceTime + NMS_Time
    fps = 1E3 / total_time
    print("success detect count:", detect_count)
    print(f"{s}Done. ({inferenceTime:.1f}ms) Inference, ({NMS_Time:.1f}ms) NMS, FPS:{fps:.1f}\n")


def bbox_filter(xyxy0, xyxy1):
    c0 = [((xyxy0[0] + xyxy0[2]) / 2), ((xyxy0[1] + xyxy0[3]) / 2)]
    c1 = [((xyxy1[0] + xyxy1[2]) / 2), ((xyxy1[1] + xyxy1[3]) / 2)]

    dis = math.sqrt(((c1[0] - c0[0])**2) + ((c1[1] - c1[1])**2))
    
    return dis <= 256


def radian_conv_degree(Radian):
    return ((Radian / math.pi) * 180)


yaw = motorCtrl(1, 0, 90)
delay(3)
pitch = motorCtrl(2, 0, 45)

def motorPID_Ctrl(frameCenter_X, frameCenter_Y):
    flag, m_flag1, m_flag2 = False, False, False  # Motor move status
    pidErr = pid.pid_run(frameCenter_X, frameCenter_Y)
    # Motor rotation
    if abs(pidErr[0]) != 0:
        yaw.incrementTurnVal(int(pidErr[0]*100))
        m_flag1 = False
    else:
        m_flag1 = True

    if abs(pidErr[1]) != 0:
        pitch.incrementTurnVal(int(pidErr[1]*100))
        m_flag2 = False
    else:
        m_flag2 = True

    # print(f"yaw: {pidErr[0]:.3f}, pitch: {pidErr[1]:.3f}")

    # get Encoder and angle
    global pub_img
    pub_img["motor_yaw"] = yaw.getAngle()
    pub_img["motor_pitch"] = pitch.getAngle()
    # print(f"{pub_img["motor_yaw"]}, {pub_img["motor_pitch"]}")

    if pub_img["motor_pitch"] > 0.0:
        pub_img["motor_pitch"] = abs(pub_img["motor_pitch"] + 45.0)
    elif pub_img["motor_pitch"] < 0.0:
        pub_img["motor_pitch"] = abs(pub_img["motor_pitch"] - 45.0)
    flag = m_flag1 and m_flag2
    return flag


def PID(xyxy):
    if xyxy is not None:
        # Calculate the center point of the image frame
        return motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(), ((xyxy[1] + xyxy[3]) / 2).item())
    return False


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
            self.img.motor_yaw, self.img.target_latitude, self.img.target_longitude, self.img.hold_status, self.img.send_info = pub_img.values()

        self.imgPublish.publish(self.img)

    def bbox_callback(self):
        self.bbox.x0 = int(pub_bbox['x0'])
        self.bbox.y0 = int(pub_bbox['y0'])
        self.bbox.x1 = int(pub_bbox['x1'])
        self.bbox.y1 = int(pub_bbox['y1'])
        self.bboxPublish.publish(self.bbox)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        # self.subscription = self.create_subscription(Img,"topic",self.holdcb,10)
        self.GlobalPositionSuub = self.create_subscription(NavSatFix, "mavros/global_position/global", self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, "mavros/imu/data", self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, "img", self.holdcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.gimbalRemove = self.create_subscription(GimbalDegree, "gimDeg", self.gimAngDegcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.hold = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.gimbalYaw = 0.0
        self.gimbalPitch = 0.0

    def gimAngDegcb(self, msg):
        self.gimbalYaw = msg.yaw
        self.gimbalPitch = msg.pitch
    
    def holdcb(self, msg):
        self.hold = pub_img["hold_status"] = msg.hold_status

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
        return pub_img["hold_status"]

    def getLatitude(self):
        return self.latitude

    def getLongitude(self):
        return self.longitude

    def getAltitude(self):
        return self.gps_altitude


pub = MinimalPublisher()
sub = MinimalSubscriber()


def _spinThread(pub, sub):
    executor = MultiThreadedExecutor()
    executor.add_node(pub)
    executor.add_node(sub)
    executor.spin()


def update_position_data():
    position.update(
        longitude=sub.getLongitude(),
        latitude=sub.getLatitude(),
        altitude=sub.getAltitude(),
        imuRoll=sub.getImuRoll(),
        imuPitch=sub.getImuPitch(),
        imuYaw=sub.getImuYaw(),
        motorYaw=pub_img["motor_yaw"],
        motorPitch=pub_img["motor_pitch"]
    )
    pub_img["second_detect"] = True


def secondDetect():
    print("second detect and drone is Hold")
    pub_img["send_info"] = True

    update_position_data()  # Update GPS, IMU, Gimbal data

    tla, tlo = position.groundTargetPostion()
    pub_img["target_longitude"], pub_img["target_latitude"], pub_img["third_detect"], pub_img["camera_center"] = tlo, tla, True, False
    delay(2)

def firstDetect():
    pub_img["send_info"] = pub_img["second_detect"] = True
    
    print("camera_center = False")    
    pub_img["target_longitude"], pub_img["target_latitude"], pub_img["camera_center"] = sub.getLongitude(), sub.getLatitude(), False
    
    update_position_data() # Update GPS, IMU, Gimbal data
    delay(2) # Delay 2s
    
    
def detect(weights, source, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, nosave=False, classes=None, agnostic_nms=False, augment=False, \
    project='runs/detect', name='exp', exist_ok=False, no_trace=False):

    source, weights, view_img, imgsz, trace = source, weights, view_img, img_size, not no_trace
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16


    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    sequentialHits = 0  # The number of consecutive target detections
    sequentialHits_status = 0
    bbox_filter_status = False
    
    # record xyxy position
    xyxy_previous = [0,0,0,0]
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() # uint8 to fp16
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # Status setting
            max_conf = -1  # Variable to store the maximum confidence value
            max_xyxy = None  # Variable to store the xyxy with the maximum confidence
            detectFlag = False
            
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            
            if len(det):
                detectFlag = True
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if conf > max_conf:
                        max_conf = conf
                        max_xyxy = xyxy
                    
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    
                
                xyxy_current = np.array([t.item() for t in max_xyxy], dtype='i4')
                
                # Tracking and bbox enable condition
                if sequentialHits > 4:
                    bbox_filter_status = bbox_filter(xyxy_current, xyxy_previous)
                    if  bbox_filter_status:
                        pub_bbox["x0"], pub_bbox['y0'], pub_bbox['x1'], pub_bbox["y1"] = xyxy_current
                        print("bbox_filter_status is True")
                
                xyxy_previous = xyxy_current.copy()
                print(f"current:{xyxy_current}, previous:{xyxy_previous}")
            else:
                pub_bbox["x0"] = pub_bbox['y0'] = pub_bbox['x1'] = pub_bbox["y1"] = 0
                
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
        sequentialHits = sequentialHits + 1 if detectFlag else 0
        sequentialHitsStatus = sequentialHits > 4
        
        print_detection_info(s, sequentialHits, t2, t1, t3)
        
        # tracking
        if bbox_filter_status:
            pub_img["camera_center"] = PID(max_xyxy)
        

def positionTask(hitsStatus:queue.Queue):
    while True:
        stat = hitsStatus.full()
        if stat:
            hs = hitsStatus.get()
            if pub_img["second_detect"] == True and pub_img["hold_status"]:
                print("second detect")
                # Continue detection after descending five meters
                if sequentialHits_status == 1 and hs:  # Detect target for the second time
                    if pub_img["camera_center"] :
                        secondDetect()           
                        while not pub_img["hold_status"]:
                            pub_img["send_info"] = False
                            # sequentialHits_status = 2

            else:  # first_detect == False
                # Target detected for the first time and Aim at targets
                if sequentialHits_status == 0 and hs:  # Detect target for the first time
                    pub_img["first_detect"] = True
                    print("first detect")

                if pub_img["camera_center"] and pub_img["hold_status"]:  # target centered and drone is hold
                    firstDetect()
                    while not sub.getHold():
                        pub_img["send_info"] = False
                        sequentialHits_status = 1


def groundGimbalCtrl(angleQ:queue.Queue):
    while True:
        if angleQ.full:
            yawA, pitchA = angleQ.get()
            yawA = int(yawA * 100)
            pitchA = int(pitchA * 100)
            yaw.singleTurnVal(yawA)
            pitch.singleTurnVal(pitchA)
        time.sleep(0.01)
    

def main():
    # ROS2
    spinThread = threading.Thread(target=_spinThread, args=(pub, sub))
    spinThread.start()
    
    # Ground Gimbal Ctrl
    degQueue = queue.Queue(maxsize=1)
    groundCtrl = threading.Thread(target=groundGimbalCtrl, args=(degQueue))
    groundCtrl.start()
    
    # Position Task
    hitsStatus = queue.Queue()
    posTask = threading.Thread(target=positionTask, args=(hitsStatus))
    posTask.start

    # Settings directly specified here
    weights = 'landpad20240522.pt'             # Model weights file path
    source = 'rtsp://127.0.0.2:8080/test'       # Data source path
    img_size = 640                    # Image size for inference
    conf_thres = 0.25                 # Object confidence threshold
    iou_thres = 0.45                  # IOU threshold for NMS
    device = '0'                       # Device to run the inference on, '' for auto-select
    view_img = not True                   # Whether to display images during processing
    nosave = False                    # Whether not to save images/videos
    # Specific classes to detect, None means detect all classes
    classes = None
    agnostic_nms = False              # Apply class-agnostic NMS
    augment = False                   # Augmented inference
    project = 'runs/detect'           # Base directory for saving runs
    name = 'exp'                      # Name of the run
    exist_ok = False                   # Overwrite existing files/directories if necessary
    no_trace = False                   # Don't trace the model for optimizations
    # Call the detect function with all the specified settings
    with torch.no_grad():
        detect(weights, source, img_size, conf_thres, iou_thres, device, view_img,
               nosave, classes, agnostic_nms, augment, project, name, exist_ok, no_trace)


if __name__ == '__main__':
    main()
