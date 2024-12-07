import time, os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ctrl.gimbal_PID import GimbalTimerTask, yaw, pitch
from parameter import Parameters
from utils_funtions import Write, increment_path
import threading as thrd
import signal
import queue, math

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
log_path = "/home/ubuntu/torch_v2/yolo_tracking_v2/detect/runs/detect"
log_save_dir = Path(increment_path(Path(log_path) / "exp", exist_ok=False))
global log
log = Write(f"{log_save_dir}/detect.txt")


def signal_handler(sig, frame):
    global yaw, pitch, executor, log
    print('Signal detected, shutting down gracefully')
    log.update_summary()
    yaw.stop()
    pitch.stop()
    executor.shutdown()
    rclpy.shutdown()
    sys.exit(0)
    

def writeToFile(filename, data):
    try:
        with open(filename, 'a') as file:
            file.write(f"{data}\n")
    except IOError as e:
        print(f"Failed to write to file: {e}")


rclpy.init(args=None)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.GlobalPositionSuub = self.create_subscription(NavSatFix, "mavros/global_position/global", self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(Imu, "mavros/imu/data", self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, "img", self.holdcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # self.gimbalRemove = self.create_subscription(GimbalDegree, "gimDeg", self.gimAngDegcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.distance = self.create_subscription(Lidar, "lidar", self.lidarcb, 10)
        self.bboxPredcd = self.create_subscription(Bbox, 'bbox', self.bboxcb, 10)
        self.motorcb = self.create_subscription(MotorInfo, 'motor_info', self.motorInfocb, 10)
        
        self.hold = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.drone_yaw = 0.0
        self.gimbalYaw = 0.0
        self.gimbalPitch = 0.0
        self.discm = 0.0
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
        self.drone_pithch = math.degrees(ned_euler_data[0])
        self.drone_roll = math.degrees(ned_euler_data[1])
        self.drone_yaw = math.degrees(ned_euler_data[2])

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
        pub_img['camera_center'] = gimbalTask.center_status
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
    
    
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    if view_img:
        view_img = check_imshow()
    
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    previous_xyxy = None
    detection_count = 0
    detect_status = False
    effective_detection = False
    
    log.write(f"source: {source}\n") # first line
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

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
            
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count


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
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)   
                                             
                # Calculate the distance between the current detection frame and the previous one
                if previous_xyxy is not None:
                    # Check whether the previous and next frames are continuous
                    ret, distance = bbox_filter(previous_xyxy, max_xyxy) 
                    # If the detection box passes the filter (ret is True), increment the detection count
                    # Otherwise, reset the detection count to 0
                    detection_count = detection_count + 1 if ret else 0
                else:
                    detection_count = 1
                    
                detect_status = True
                effective_detection = detection_count >= 4
                pub_img['detect'] = pub_bbox['detect'] = effective_detection
                previous_xyxy = max_xyxy   
            else:
                detect_status = False
                pub_img['detect'] = pub_bbox['detect'] = False
            
            # Print time (inference + NMS) and gimbal Degrees
            fps = 1E3/((t3-t1)*1E3)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, FPS:{fps:.1f}')
            print(f"Total Pitch Degrees: {pub_img['motor_pitch']:.2f}, yaw Degrees: {pub_img['motor_yaw']:.2f}")
            
            # Publish the detection results
            if max_xyxy is not None and effective_detection:
                Update_pub_bbox(effective_detection, n, max_conf, max_xyxy[0], max_xyxy[1], max_xyxy[2], max_xyxy[3])
                effective_detection = f"Name: {names[int(cls)]}, Conf: {max_conf}, Bbox: {max_xyxy}"
            else:
                Update_pub_bbox(False, 0, 0.0, 1280, 720)
            
            # Log the detection results
            log.log_detection_result(detect_status, detection_count, effective_detection)
            
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                
def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
     
    # ROS
    global ROS_Pub, ROS_Sub

    ROS_spin = thrd.Thread(target=_spinThread, args=(ROS_Pub, ROS_Sub, gimbalTask))
    ROS_spin.start()
    
    # YOLO
    # Settings directly specified here
    weights = 'landpad20240522.pt'                                              # Model weights file path
    source ='rtsp://127.0.0.' + str(np.random.randint(1,256)) + ':8080/video_feed'    # Data source path
    # Data source path
    img_size = 640                                                              # Image size for inference
    conf_thres = 0.4                                                            # Object confidence threshold
    iou_thres = 0.3                                                             # IOU threshold for NMS
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
