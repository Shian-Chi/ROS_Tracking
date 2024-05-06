import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import NavSatFix, Imu
from mavros_msgs.msg import Altitude
from transforms3d import euler
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel
import threading
import queue
from tutorial_interfaces.msg import Img
from pid.pid import PID_Ctrl
from pid.parameter import Parameters
from pid.motor import yaw, pitch
from pid.position import verticalTargetPositioning
import os
import time
import math

# Set the DISPLAY environment variable globally
# os.environ['DISPLAY'] = ':0'

hold = False
frame = None

pub_img = {'first_detect': False,
           'second_detect': False,
           'third_detect': False,
           'camera_center': False,
           'motor_pitch': 0.0,
           'motor_yaw': 0.0,
           'target_latitude': 0.0,
           'target_longitude': 0.0
           }


para = Parameters()
pid = PID_Ctrl()


def delay(s):
    s += 1
    for i in range(1, s):
        print(f'{i}s')
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
    _, pub_img['motor_yaw'] = yaw.getEncoderAndAngle()
    _, pub_img['motor_pitch'] = pitch.getEncoderAndAngle()
    # print(f"{pub_img['motor_yaw']}, {pub_img['motor_pitch']}")
    
    if pub_img['motor_pitch'] > 0.0:
        pub_img['motor_pitch'] = abs(pub_img['motor_pitch'] + 45.0)
    elif pub_img['motor_pitch'] < 0.0:
        pub_img['motor_pitch'] = abs(pub_img['motor_pitch'] - 45.0)
    flag = m_flag1 and m_flag2
    return flag


def PID(xyxy):
    if xyxy is not None:
        # Calculate the center point of the image frame
        return motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(), ((xyxy[1] + xyxy[3]) / 2).item())
    return False


class RTSPCamera():
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
        self.output_queue = queue.Queue(maxsize=1)
        self.frame = None
        self.running = True
        self.read_thread = threading.Thread(target=self.read_rtsp)
        self._img_get = threading.Thread(target=self.image_get)

    def read_rtsp(self):
        if not self.cap.isOpened():
            print("Error: Unable to open camera.")
            return

        print("Opened camera")
        try:
            while self.running:
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                # frame = cv2.resize(frame, (1280,720))
                if not ret:
                    print("Error: Unable to read frame.")
                    break

                self.output_queue.put(frame)
        except Exception as err:
            print("Camera processing", err)

    def image_get(self):
        global frame
        while True:
            frame = self.output_queue.get()

    def start(self):
        self.read_thread.start()
        self._img_get.start()

    def close(self):
        self.cap.release()


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.imgPublish = self.create_publisher(Img, 'img', 10)
        timer_period = 0.1
        self.img_timer = self.create_timer(timer_period, self.img_callback)

    def img_callback(self):
        img = Img()

        img.first_detect, img.second_detect, img.third_detect, img.camera_center, img.motor_pitch, \
            img.motor_yaw, img.target_latitude, img.target_longitude = pub_img.values()

        self.imgPublish.publish(img)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # self.subscription = self.create_subscription(Img,'topic',self.holdcb,10)
        self.GlobalPositionSuub = self.create_subscription(
            NavSatFix, 'mavros/global_position/global', self.GPcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imuSub = self.create_subscription(
            Imu, 'mavros/imu/data', self.IMUcb, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.holdSub = self.create_subscription(Img, 'img', self.holdcb, QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.hold = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.gps_altitude = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0

    def holdcb(self, msg):
        self.hold = msg.hold_status

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
        return self.hold

    def getLatitude(self):
        return self.latitude

    def getLongitude(self):
        return self.longitude

    def getAltitude(self):
        return self.gps_altitude


class YOLO():
    def __init__(self, weight, cap, sub):
        self.weights = weight
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.agnostic = False
        self.augment = False

        self.cap = cap
        self.sub = sub
        self.hold = False

        self.detectFlag = False

        self.position = verticalTargetPositioning()

        # Initialize
        set_logging()
        try:
            self.device = select_device('0')
        except:
            self.device = select_device('cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        
        self.img, self.im0s = [None], None
        self.img0 = None
        self.im0 = None
        self.imgs = [None] * 1        
        self.model.eval()
        if self.half:
            self.model.half()  # to FP16

        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255)for _ in range(3)] for _ in self.names]

        # Thread
        im_show = threading.Thread(target=self.show)
        im_show.start()
                
    def print_detection_info(self, s, detect_count, end_inference, start_inference, end_nms):
        inferenceTime = 1E3 * (end_inference - start_inference)
        NMS_Time = 1E3 * (end_nms - end_inference)
        total_time = inferenceTime + NMS_Time
        fps = 1E3 / total_time
        # print("success detect count:", detect_count)
        print(f'{s}Done. ({inferenceTime:.1f}ms) Inference, ({NMS_Time:.1f}ms) NMS, FPS:{fps:.1f}\n')
    
    def show(self):
        while True:
            if self.im0 is not None:
                cv2.imshow("bbox", self.im0)
                if cv2.waitKey(1) == ord('q'):
                    break
        cv2.destroyAllWindows()
        
    def update_position_data(self):
        self.position.update(
            longitude=self.sub.getLongitude(),
            latitude=self.sub.getLatitude(),
            altitude=self.sub.getAltitude(),
            imuRoll=self.sub.getImuRoll(),
            imuPitch=self.sub.getImuPitch(),
            imuYaw=self.sub.getImuYaw(),
            motorYaw=pub_img['motor_yaw'],
            motorPitch=pub_img['motor_pitch']
        )
        pub_img['second_detect'] = True
    
    def yolo(self):
        t1 = time_synchronized()
        # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = self.model(self.img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic)
        t3 = time_synchronized()
        return t1, t2, t3, pred      
    
    def runYOLOandObjectPositioning(self):
        detect_count = 0
        count = 0
        bflag = False
        try:
            while True:
                self.loadimg()
                # Inference
                t1, t2, t3, pred = self.yolo()

                self.detectFlag = False

                for i, det in enumerate(pred):  # detections per image
                    s, self.im0 = '%g: ' % i, self.im0s[i].copy()

                    max_conf = -1  # Variable to store the maximum confidence value
                    max_xyxy = None  # Variable to store the xyxy with the maximum confidence
                    n = 0
                    if len(det):
                        # Rescale boxes from img_size to self.im0 size
                        det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], self.im0.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            # detections per class
                            n = (det[:, -1] == c).sum()
                            # add to string
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                        # Find a suitable target and plot bboxs
                        for *xyxy, conf, cls in reversed(det):
                            '''
                            # Find tracking target
                            if self.names[int(cls)] == "landing-pad": 
                                self.detectFlag = True
                                # Track your most trusted targets
                                if conf > max_conf:
                                    max_conf = conf
                                    max_xyxy = xyxy
                            '''

                            # Find the target with the highest trust level
                            if conf > max_conf:
                                max_conf = conf
                                max_xyxy = xyxy
                                self.detectFlag = True
                                
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, self.im0, label=label, color=self.colors[int(cls)], line_thickness=1)    
                                
                        # return camera image centered state
                        pub_img['camera_center'] = PID(max_xyxy)

                        if pub_img['second_detect'] == True:
                            # Continue detection after descending five meters
                            if count == 1 and detect_count > 4:  # Detect target for the second time
                                if pub_img['camera_center'] and self.sub.getHold():
                                    print('second detect and drone is Hold')

                                    # Update GPS, IMU, Gimbal data
                                    self.update_position_data()

                                    tla, tlo = self.position.groundTargetPostion()
                                    pub_img['target_longitude'], pub_img['target_latitude'], pub_img['third_detect'] = tlo, tla, True
                                    delay(2)
                                    pub_img['camera_center'] = False
                                else:
                                    continue
                        else:  # first_detect == False
                            # Target detected for the first time and Aim at target
                            if count == 0 and detect_count > 4:  # Detect target for the first time
                                pub_img['first_detect'] = True
                                print("first detect")
                                # not center, drone is hold
                                if not pub_img['camera_center'] and (self.sub.getHold()):
                                    print("drone is Hold")
                                    continue

                                if pub_img['camera_center']:  # center
                                    # Update GPS, IMU, Gimbal data
                                    pub_img['second_detect'] = True
                                    self.update_position_data()
                                    # Delay 2s
                                    delay(2)
                                    pub_img['camera_center'] = False
                                    count = 1                
                    
                    detect_count = detect_count + 1 if self.detectFlag else 0

                    # Print time (inference + NMS)
                    self.print_detection_info(s, detect_count, t2, t1, t3)
                
                if bflag:
                    break
            cv2.destroyAllWindows()
                
                    
        except Exception as err:
            print("YOLO error:", err)
        except KeyboardInterrupt:
            print("Ctrl+C")
        finally:
            self.cap.close()
            self.cap.read_thread.join()
            self.cap._img_get.join()
            yaw.stop()
            pitch.stop()
            exit(1)

    def loadimg(self):
        global frame

        '''self.im0s = frame.copy()

        # Letterbox
        self.img = letterbox(self.im0s, 640, auto=True, stride=32)[0]

        if not isinstance(self.img, np.ndarray):
            self.img = np.array(self.img)
        
        self.img = np.expand_dims(self.img, axis=0)

        # Stack
        # self.img = np.stack(self.img, 0)
        self.img = torch.from_numpy(self.img)'''
        
        self.imgs[0] = frame

        self.im0s = self.imgs.copy()

        # Letterbox
        self.img = [letterbox(x, 640, auto=True, stride=32)[0] for x in self.im0s]

        # Stack
        self.img = np.stack(self.img, 0)
        self.img = torch.from_numpy(self.img)
        
        # Convert
        # BGR to RGB, to bsx3x416x416
        # print(self.img.shape)
        self.img = self.img.permute(0, 3, 1, 2)
        self.img = self.img.contiguous().to(self.device)

        self.img = self.img.half() if self.half else self.img.float()  # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)

    def trackingStart(self):
        self.runYOLOandObjectPositioning()



def _spinThread(pub, sub):
    executor = MultiThreadedExecutor()
    executor.add_node(pub)
    executor.add_node(sub)
    executor.spin()


def main(args=None):
    rclpy.init(args=args)

    publisher = MinimalPublisher()
    subscriber = MinimalSubscriber()
    spinThread = threading.Thread(
        target=_spinThread, args=(publisher, subscriber))
    try:
        spinThread.start()

        cap = RTSPCamera("rtsp://0.0.0.0:8080/test")
        cap.start()

        yolo = YOLO("landpad20140411.pt", cap, subscriber)
        yolo.trackingStart()
    except KeyboardInterrupt:
        yaw.stop()
        pitch.stop()
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
