import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
# from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from pid.PID_Calc import PID_Ctrl
from pid.parameter import Parameters
from pid.motor import motorCtrl

import threading as thrd
import sys, signal
import queue, math

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import NavSatFix, Imu
from transforms3d import euler
from cv_bridge import CvBridge
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
    'x1': 720,
    'y0': 0,
    'y1': 0
}


pub_motor ={
    'pitchAngle': 0.0,
    'yawAngle': 0.0,
    'pitchPluse' : 0,
    'yawPluse' : 0
}


pid = PID_Ctrl()
bridge = CvBridge()
para = Parameters()

def signal_handler(sig, frame):
    global yaw, pitch, ROS_Pub, ROS_Sub
    print('Signal detected, shutting down gracefully')
    yaw.stop()
    pitch.stop()
    ROS_Pub.destroy_node()
    ROS_Sub.destroy_node()
    rclpy.shutdown()
    sys.exit(0)
    

def radian_conv_degree(Radian):
    return ((Radian / math.pi) * 180)


def writeToFile(filename, data):
    try:
        with open(filename, 'a') as file:
            file.write(f"{data}\n")
    except IOError as e:
        print(f"Failed to write to file: {e}")


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        # Img publish
        self.imgPublish = self.create_publisher(Img, "img", 10)
        img_timer_period = 1/35
        self.img_timer = self.create_timer(img_timer_period, self.img_callback)
        
        # Bbox publish
        self.bboxPublish = self.create_publisher(Bbox, "bbox", 10)
        bbox_timer_period = 1/10
        self.img_timer = self.create_timer(bbox_timer_period, self.bbox_callback)
       
        # MotorInfo publish
        self.motorInfoPublish = self.create_publisher(MotorInfo, "motor_info", 10)
        motor_timer_period = 1/10
        self.motor_timer = self.create_timer(motor_timer_period, self.motor_callback)
        
        self.img = Img()
        
        self.bbox = Bbox()
        self.motorInfo = MotorInfo()
        
    def img_callback(self):
        self.img.detect, self.img.camera_center, self.img.motor_pitch, self.img.motor_yaw, \
            self.img.target_latitude, self.img.target_longitude, self.img.hold_status, self.img.send_info = pub_img.values()        
        self.imgPublish.publish(self.img)
    
    def bbox_callback(self):
        bbox_msg = Bbox()
        bbox_msg.class_id = pub_bbox['class_id']
        bbox_msg.confidence = pub_bbox['confidence']

        bbox_msg.x0 = pub_bbox['x0']
        bbox_msg.y0 = pub_bbox['y0']

        bbox_msg.x1 = pub_bbox['x1']
        bbox_msg.y1 = pub_bbox['y1']

        # Publish BoundingBox message
        self.bboxPublish.publish(bbox_msg)

    def motor_callback(self):
        _, yawData = yaw.getEncoder()
        time.sleep(0.01)
        _, pitchData = pitch.getEncoder()
        
        self.motorInfo.pitch_pluse = pub_motor['pitchPluse'] = pitchData
        self.motorInfo.yaw_pluse =   pub_motor['yawPluse'] = yawData  
        self.motorInfo.pitch_angle = pub_motor['pitchAngle']  = pitchData / para.uintDegreeEncoder
        self.motorInfo.yaw_angle =   pub_motor['yawAngle'] = yawData / para.uintDegreeEncoder
        
        self.motorInfoPublish.publish(self.motorInfo)
    
    
class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.GlobalPositionSuub = self.create_subscription(NavSatFix, "mavros/global_position/global", self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, "mavros/imu/data", self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, "img", self.holdcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # self.gimbalRemove = self.create_subscription(GimbalDegree, "gimDeg", self.gimAngDegcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.distance = self.create_subscription(Lidar, "lidar", self.lidarcb, 10)
        self.hold = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.gimbalYaw = 0.0
        self.gimbalPitch = 0.0
        self.discm = 0.0

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
        self.pithch = radian_conv_degree(ned_euler_data[0])
        self.roll = radian_conv_degree(ned_euler_data[1])
        self.yaw = radian_conv_degree(ned_euler_data[2])

    def lidarcb(self, msg):
        self.discm = msg.distance_cm
    
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

    def getDistance(self):
        return self.discm


def _spinThread(pub, sub):
    # Create an executor and spin the ROS nodes in this process
    executor = MultiThreadedExecutor()
    executor.add_node(pub)
    executor.add_node(sub)

    try:
        executor.spin()
    finally:
        # Shutdown the nodes after spinning
        pub.destroy_node()
        sub.destroy_node()
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


def motorPID_Ctrl(frameCenter_X, frameCenter_Y):
    m_flag1, m_flag2 = False, False  # Motor move status
    pidErr = pid.pid_run(frameCenter_X, frameCenter_Y)
    
    # Motor rotation
    if abs(pidErr[0]) != 0:
        yaw.incrementTurnVal(int(pidErr[0]*100))
    else:
        m_flag1 = True

    if abs(pidErr[1]) != 0:
        pitch.incrementTurnVal(int(pidErr[1]*100))
    else:
        m_flag2 = True

    return m_flag1 and m_flag2, pidErr[0], pidErr[1]


def PID(xyxy):
    global pub_img
    if (xyxy is not None) and pub_img['detect']:
        # Calculate the center point of the image frame
        return motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(), ((xyxy[1] + xyxy[3]) / 2).item())
    return False, 0.0, 0.0


def getGimbalEncoders():
    Y_ret, Y_Encoder= yaw.getEncoder()
    P_ret, P_Encoder= pitch.getEncoder()
    return Y_Encoder, P_Encoder
            
            
def gimbalCtrl(xyxyCtx): 
    global pub_img
    p = "/home/ubuntu/yolo/yolo_tracking_v2/gimbalAngle/angle.txt"
    camera_center = False
    while True:
        if xyxyCtx.full:
            camera_center, Y_pidErr, P_pidErr, = PID(xyxyCtx.get())
            
            print(f"camera_center: {camera_center},\nYawError: {Y_pidErr}, PitchError: {P_pidErr}")
            if camera_center is True:
                y, p = getGimbalEncoders()
                pub_img['camera_center'] = camera_center
                
                pub_motor['yawPluse'], pub_motor['pitchAngle'] = y, p
                pub_motor['yawAngle'], pub_motor['pitchAngle'] = y / para.uintDegreeEncoder, p / para.uintDegreeEncoder
                
                pub_img['motor_pitch'] = pub_motor['pitchAngle']
                pub_img['motor_yaw'] = pub_motor['yawAngle']
                
                print(f"yaw angle: {pub_motor['yawAngle']}, pitch angle: {pub_motor['pitchAngle']}")
            else:
                pub_img['camera_center'] = False
                

def bbox_filter(xyxy0, xyxy1):
    c0 = [((xyxy0[0] + xyxy0[2]) / 2), ((xyxy0[1] + xyxy0[3]) / 2)]
    c1 = [((xyxy1[0] + xyxy1[2]) / 2), ((xyxy1[1] + xyxy1[3]) / 2)]
    
    dis = math.sqrt(((c1[0] - c0[0])**2) + ((c1[1] - c0[1])**2))
    return dis<=256, dis
        

def detect(weights, source, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, classes=None, agnostic_nms=False, augment=False, no_trace=False):
    source, weights, view_img, imgsz, trace = source, weights, view_img, img_size, not no_trace
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

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    if view_img:
        view_img = check_imshow()
        view_img = True
    
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    previous_xyxy = None
    detection_count = 0
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b, old_img_h, old_img_w = img.shape[0], img.shape[2], img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
                          
        # Process detections
        global pub_img, pub_bbox
        for i, det in enumerate(pred):  # detections per image
            # Status setting
            n = 0 # Classifier
            max_conf = -1  # Variable to store the maximum confidence value
            max_xyxy = None  # Variable to store the xyxy with the maximum confidence
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string                   

                for *xyxy, conf, cls in reversed(det):
                    if conf > max_conf:
                        max_conf, max_xyxy = conf, xyxy
                    
                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3) # im0 type: <class 'numpy.ndarray'>
                        
                # Calculate the distance between the current detection frame and the previous one
                if previous_xyxy is not None:
                    # Check whether the previous and next frames are continuous
                    ret, distance = bbox_filter(previous_xyxy, max_xyxy) 
                    detection_count = detection_count + 1 if ret else 0
                else:
                    detection_count = 1
                
                detect_status = detection_count >= 4
                pub_img['detect'] = pub_bbox['detect'] = detect_status
                
                if detect_status:
                    xyxyCtx.put(max_xyxy)
                previous_xyxy = max_xyxy
                
            else:
                pub_img['detect'] = pub_bbox['detect'] = False
            
            if max_xyxy is not None:
                Update_pub_bbox(detect_status, n, max_conf, max_xyxy[0], max_xyxy[1], max_xyxy[2], max_xyxy[3])
            else:
                Update_pub_bbox(False, 0, 0.0, 1280, 720)
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, FPS:{1E3/((t3-t1)*1E3):.1f}')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        
def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
  
    # Gimbal
    global xyxyCtx
    global yaw, pitch
    yaw = motorCtrl(1, "yaw", 0, 90.0)
    pitch = motorCtrl(2, "pitch", 0, 360.0)
    print(f"\033[34m Yaw encoder: {yaw.info.encoder},\n Pitch encoder:{pitch.info.encoder} \033[m")
    xyxyCtx = queue.Queue()
    gimbalCtrlThread = thrd.Thread(target=gimbalCtrl, args=(xyxyCtx,))
    gimbalCtrlThread.start()
    
    # ROS
    rclpy.init(args=args)
    global ROS_Pub, ROS_Sub
    ROS_Pub = MinimalPublisher()
    ROS_Sub = MinimalSubscriber()
    ROS_spin = thrd.Thread(target=_spinThread, args=(ROS_Pub, ROS_Sub))
    ROS_spin.start()
    
    # YOLO
    # Settings directly specified here
    weights = 'landpad20240522.pt'                                              # Model weights file path
    source ='rtsp://127.0.0.' + str(np.random.randint(1,256)) + ':8080/test'    # Data source path
    # Data source path
    img_size = 640                                                              # Image size for inference
    conf_thres = 0.45                                                           # Object confidence threshold
    iou_thres = 0.3                                                            # IOU threshold for NMS
    device = '0'                                                                # Device to run the inference on, '' for auto-select
    view_img = not True                                                         # Whether to display images during processing
    # Specific classes to detect, None means detect all classes
    classes = None
    agnostic_nms = False                                                        # Apply class-agnostic NMS
    augment = False                                                             # Augmented inference
    no_trace = False                                                            # Don't trace the model for optimizations
    # Call the detect function with all the specified settings
    with torch.no_grad():
        detect(weights, source, img_size, conf_thres, iou_thres, device, view_img,
               classes, agnostic_nms, augment, no_trace)


if __name__ == '__main__':
    main()
