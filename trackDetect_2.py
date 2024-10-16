import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import NavSatFix, Imu
from mavros_msgs.msg import Altitude
from transforms3d import euler
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import threading
import queue
from tutorial_interfaces.msg import Img
from pid.pid import PID_Ctrl
from pid.parameter import Parameters
from pid.motor import yaw, pitch
from pid.position import verticalTargetPositioning
import math

hold = False
frame = None

pub_img = {"first_detect": False,
           "second_detect": False,
           "third_detect": False,
           "camera_center": False,
           "motor_pitch": 0.0,
           "motor_yaw": 0.0,
           "target_latitude": 0.0,
           "target_longitude": 0.0,
           "hold_status": False,
           "send_info": False
           }


para = Parameters()
pid = PID_Ctrl()


def delay(s: int):
    s += 1
    for i in range(1, s):
        print(f"{i}s")
        time.sleep(1)


def radian_conv_degree(Radian):
    return ((Radian / math.pi) * 180)


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

    print(f"yaw: {pidErr[0]:.3f}, pitch: {pidErr[1]:.3f}")

    # get Encoder and angle
    global pub_img
    _, pub_img["motor_yaw"] = yaw.getEncoderAndAngle()
    _, pub_img["motor_pitch"] = pitch.getEncoderAndAngle()
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


rclpy.init()


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        self.imgPublish = self.create_publisher(Img, "img", 10)
        timer_period = 1/35
        self.img_timer = self.create_timer(timer_period, self.img_callback)

        self.img = Img()
        
    def img_callback(self):
        self.img.first_detect, self.img.second_detect, self.img.third_detect, self.img.camera_center, self.img.motor_pitch, \
            self.img.motor_yaw, self.img.target_latitude, self.img.target_longitude, self.img.hold_status, self.img.send_info = pub_img.values()

        self.imgPublish.publish(self.img)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        # self.subscription = self.create_subscription(Img,"topic",self.holdcb,10)
        self.GlobalPositionSuub = self.create_subscription(
            NavSatFix, "mavros/global_position/global", self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(
            Imu, "mavros/imu/data", self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, "img", self.holdcb, QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.hold = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0

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


position = verticalTargetPositioning()


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


def print_detection_info(s, detect_count, end_inference, start_inference, end_nms):
    inferenceTime = 1E3 * (end_inference - start_inference)
    NMS_Time = 1E3 * (end_nms - end_inference)
    total_time = inferenceTime + NMS_Time
    fps = 1E3 / total_time
    print("success detect count:", detect_count)
    print(f"{s}Done. ({inferenceTime:.1f}ms) Inference, ({NMS_Time:.1f}ms) NMS, FPS:{fps:.1f}\n")


def detect(weights, source, img_size=640, conf_thres=0.25, iou_thres=0.45, device="", view_img=False, classes=None, agnostic_nms=False, augment=False, no_trace=False,
    project='runs/detect', name='exp', exist_ok=False, nosave=False):

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if no_trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    # Set Dataloader
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, stride=stride)
#    else:
#        dataset = LoadImages(source, img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, img_size, img_size).to(
            device).type_as(next(model.parameters())))  # run once

    sequentialHits = 0  # The number of consecutive target detections
    sequentialHits_status = 0

    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup and inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Status setting
        max_conf = -1  # Variable to store the maximum confidence value
        max_xyxy = None  # Variable to store the xyxy with the maximum confidence
        detectFlag = False

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, vid_cap = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, vid_cap = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # normalization gain whwh
            if len(det):
                detectFlag = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

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
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(xyxy, im0, label=label,color=colors[int(cls)], line_thickness=1)

            # # Stream results
#            if view_img:
#                cv2.imshow(str(p), im0)
#                cv2.waitKey(1)  # 1 millisecond
                        # Save results (image with detections)
            if save_img:
                if dataset.mode == 'stream' or dataset.mode == 'video':
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(30)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        print_detection_info(s, sequentialHits, t2, t1, t3)

        sequentialHits = sequentialHits + 1 if detectFlag else 0
        sequentialHitsStatus = sequentialHits > 4
        
        # tracking
        pub_img["camera_center"] = PID(max_xyxy)
        
        if pub_img["second_detect"] == True and pub_img["hold_status"]:
            print("second detect")
            # Continue detection after descending five meters
            if sequentialHits_status == 1 and sequentialHitsStatus:  # Detect target for the second time
                if pub_img["camera_center"] :
                    print("second detect and drone is Hold")
                    pub_img["send_info"] = True

                    # Update GPS, IMU, Gimbal data
                    update_position_data()

                    tla, tlo = position.groundTargetPostion()
                    pub_img["target_longitude"], pub_img["target_latitude"], pub_img["third_detect"] = tlo, tla, True
                    pub_img["camera_center"] = False
                    delay(2)
                    
                    while not pub_img["hold_status"]:
                        pub_img["send_info"] = False
                        sequentialHits_status = 2

        else:  # first_detect == False
            # Target detected for the first time and Aim at targets
            if sequentialHits_status == 0 and sequentialHitsStatus:  # Detect target for the first time
                pub_img["first_detect"] = True
                print("first detect")

            if pub_img["camera_center"] and pub_img["hold_status"]:  # target centered and drone is hold
                pub_img["send_info"] = pub_img["second_detect"] = True
                
                print("send drone position")
                pub_img["target_longitude"], pub_img["target_latitude"] = sub.getLongitude(), sub.getLatitude()
                
                print("camera_center = False")
                pub_img["camera_center"] = False
                
                # Update GPS, IMU, Gimbal data
                update_position_data()

                # Delay 2s
                delay(2)
                while not sub.getHold():
                    pub_img["hold_status"] = pub_img["send_info"] = False
                    sequentialHits_status = 1


def main():
    try:
        # ROS2
        spinThread = threading.Thread(target=_spinThread, args=(pub, sub))
        spinThread.start()
        # Settings directly specified here
        weights = "landpad20140411.pt"             # Model weights file path
        source = "rtsp://0.0.0.0:8080/test"       # Data source path
        img_size = 640                    # Image size for inference
        conf_thres = 0.5                 # Object confidence threshold
        iou_thres = 0.45                  # IOU threshold for NMS
        device = ""                       # Device to run the inference on, "" for auto-select
        view_img = True                   # Whether to display images during processing
        # Specific classes to detect, None means detect all classes
        classes = None
        agnostic_nms = False              # Apply class-agnostic NMS
        augment = False                   # Augmented inference
        no_trace = False                   # Don't trace the model for optimizations

        project = 'runs/detect'           # Base directory for saving runs
        name = 'exp'
        exist_ok = True                   # Overwrite existing files/directories if necessary
        nosave = False                    # Whether not to save images/videos

        # Call the detect function with all the specified settings
        with torch.no_grad():
            detect(weights = weights, 
                   source = source,
                   img_size = img_size, 
                   conf_thres = conf_thres, iou_thres=iou_thres, 
                   device = device, 
                   view_img = view_img, 
                   classes = classes, 
                   agnostic_nms = agnostic_nms, 
                   augment = augment, 
                   no_trace = no_trace, 
                   project = project, 
                   name = name,
                   exist_ok = exist_ok, 
                   nosave = nosave)

    except KeyboardInterrupt:
        yaw.stop()
        pitch.stop()
        pub.destroy_node()
        sub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()