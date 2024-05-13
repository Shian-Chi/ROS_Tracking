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

from gimbal.pid import PID_Ctrl
from gimbal.parameter import Parameters
from gimbal.motor import motorCtrl
from gimbal.position import verticalTargetPositioning
from gimbal.timer import timer
from ros.pub_sub import pub, sub, _spinThread
from ros.ros_parameter import pub_bbox, pub_img
import threading

para = Parameters()
pid = PID_Ctrl()
position = verticalTargetPositioning()
pubImgData = pub_img
pubBboxData = pub_bbox

sequentialHits_status = 0
sequentialHitsStatus = False

runTime = None
timer_1 = timer()

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


def gimbal_Init():
    global yaw, pitch
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
    """    
    global pubImgData
    pubImgData["motor_yaw"] = yaw.getAngle()
    pubImgData["motor_pitch"] = pitch.getAngle()
    # print(f"{pubImgData["motor_yaw"]}, {pubImgData["motor_pitch"]}")

    if pubImgData["motor_pitch"] > 0.0:
        pubImgData["motor_pitch"] = abs(pubImgData["motor_pitch"] + 45.0)
    elif pubImgData["motor_pitch"] < 0.0:
        pubImgData["motor_pitch"] = abs(pubImgData["motor_pitch"] - 45.0)
    """
    flag = m_flag1 and m_flag2
    
    return flag


def PID(xyxy):
    if xyxy is not None:
        # Calculate the center point of the image frame
        return motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(), ((xyxy[1] + xyxy[3]) / 2).item())
    return False


def update_position_data():
    position.update(
        longitude=sub.getLongitude(),
        latitude=sub.getLatitude(),
        altitude=sub.getAltitude(),
        imuRoll=sub.getImuRoll(),
        imuPitch=sub.getImuPitch(),
        imuYaw=sub.getImuYaw(),
        motorYaw=pubImgData["motor_yaw"],
        motorPitch=pubImgData["motor_pitch"]
    )
    pubImgData["second_detect"] = True


def secondDetect():
    print("second detect and drone is Hold")
    pubImgData["send_info"] = True

    update_position_data()  # Update GPS, IMU, Gimbal data

    tla, tlo = position.groundTargetPostion()
    pubImgData["target_longitude"], pubImgData["target_latitude"], pubImgData["third_detect"], pubImgData["camera_center"] = tlo, tla, True, False
    delay(2)


def firstDetect():
    pubImgData["send_info"] = pubImgData["second_detect"] = True

    print("camera_center = False")
    pubImgData["target_longitude"], pubImgData["target_latitude"], pubImgData["camera_center"] = sub.getLongitude(), sub.getLatitude(), False

    update_position_data()  # Update GPS, IMU, Gimbal data
    delay(2)  # Delay 2s


def setDataStatus(HitsStatus):
    if pubImgData["second_detect"] == True and pubImgData["hold_status"]:
        print("second detect")
        # Continue detection after descending five meters
        if sequentialHits_status == 1 and HitsStatus:  # Detect target for the second time
            if pubImgData["camera_center"]:
                secondDetect()
                while not pubImgData["hold_status"]:
                    pubImgData["send_info"] = False
                    sequentialHits_status = 2

    else:  # first_detect == False
        # Target detected for the first time and Aim at targets
        if sequentialHits_status == 0 and HitsStatus:  # Detect target for the first time
            pubImgData["first_detect"] = True
            print("first detect")

        # target centered and drone is hold
        if pubImgData["camera_center"] and pubImgData["hold_status"]:
            firstDetect()
            while not sub.getHold():
                pubImgData["send_info"] = False
                sequentialHits_status = 1


def distance(xyxy0, xyxy1):
    c0 = [((xyxy0[0] + xyxy0[2]) / 2), ((xyxy0[1] + xyxy0[3]) / 2)]
    c1 = [((xyxy1[0] + xyxy1[2]) / 2), ((xyxy1[1] + xyxy1[3]) / 2)]

    return math.sqrt(((c1[0] - c0[0])**2) + ((c1[1] - c1[1])**2))

def save_time(data):
    filename = "time.txt"
    with open(filename, 'a') as file:
        # 寫入一些文本行
        file.write(f"{data}\n")


def detect(weights, source, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, nosave=False, classes=None, agnostic_nms=False, augment=False,
           project='runs/detect', name='exp', exist_ok=False, no_trace=False, save_txt=False):

    source, weights, view_img, imgsz, trace = source, weights, view_img, img_size, not no_trace
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name,exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    # Set Dataloader
    vid_path, vid_writer = None, None

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    sequentialHits = 0  # The number of consecutive target detections
    
    bbox_filter_status = False

    # record xyxy position
    xyxy_previous = [0, 0, 0, 0]

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()  # uint8 to fp16
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
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            if len(det):
                runTime = f"{timer_1.start_timer()}"
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

                    if save_txt:  # Write to file
                        line = (cls, *max_xyxy, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                xyxy_current = np.array([t.item() for t in max_xyxy], dtype='i4')

                # Tracking and bbox enable condition
                """
                if sequentialHits > 4:
                    bbox_filter_status = bbox_filter(xyxy_current, xyxy_previous)
                    if  bbox_filter_status:
                        pubBboxData["x0"], pubBboxData['y0'], pubBboxData['x1'], pubBboxData["y1"] = xyxy_current
                        print("bbox_filter_status is True")
                
                xyxy_previous = xyxy_current.copy()
                print(f"current:{xyxy_current}, previous:{xyxy_previous}")
            else:
                pubBboxData["x0"] = pubBboxData['y0'] = pubBboxData['x1'] = pubBboxData["y1"] = 0
                """
                
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            """
            if save_img:
                if dataset.mode == 'stream' or dataset.mode == 'video':
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            """
        sequentialHits = sequentialHits + 1 if detectFlag else 0
        sequentialHitsStatus = sequentialHits > 4

        print_detection_info(s, sequentialHits, t2, t1, t3)

        # tracking
        # if bbox_filter_status:
        #     pubImgData["camera_center"] = PID(max_xyxy)
        pubImgData["camera_center"] = PID(max_xyxy)
        if pubImgData["camera_center"]:
            runTime += f"{timer_1.stop_timer()}"
        setDataStatus(sequentialHitsStatus)



def main():
    gimbal_Init()

    # ROS2
    spinThread = threading.Thread(target=_spinThread, args=(pub, sub))
    spinThread.start()

    # Settings directly specified here
    weights = 'landpad20140411.pt'             # Model weights file path
    source = 'rtsp://0.0.0.0:8080/test'       # Data source path
    img_size = 640                    # Image size for inference
    conf_thres = 0.25                 # Object confidence threshold
    iou_thres = 0.45                  # IOU threshold for NMS
    device = '0'                       # Device to run the inference on, '' for auto-select
    view_img = True                   # Whether to display images during processing
    nosave = False                    # Whether not to save images/videos
    # Specific classes to detect, None means detect all classes
    classes = None
    agnostic_nms = False              # Apply class-agnostic NMS
    augment = False                   # Augmented inference
    project = 'runs/detect'           # Base directory for saving runs
    name = 'exp'                      # Name of the run
    exist_ok = False                   # Overwrite existing files/directories if necessary
    no_trace = False                   # Don't trace the model for optimizations
    save_txt = False                   # Save results to runs/<project>/*.txt
    # Call the detect function with all the specified settings
    with torch.no_grad():
        detect(weights, source, img_size, conf_thres, iou_thres, device, view_img,
               nosave, classes, agnostic_nms, augment, project, name, exist_ok, no_trace)


if __name__ == '__main__':
    main()
