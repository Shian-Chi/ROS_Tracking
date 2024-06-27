import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

import torch.multiprocessing as mp
import queue
import signal
import sys

def signal_handler(signal, frame):
    print(f"Received signal: {signal}, shutting down...")
    mp_gimbal.terminate()
    mp_task.terminate()    
    sys.exit(0)
    
pub_img = {
    "first_detect": False,
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

pub_bbox = {
    'x0': 0,
    'x1': 0,
    'y0': 0,
    'y1': 0
}

def manage_queue(q:mp.Queue, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        removed = q.get()
        q.put(item)
    except queue.Empty:
        print("Attempted to remove item from an empty queue")
    except Exception as err:
        print(f"manage_queue error: {err}")

class ObjectDetector:
    def __init__(self, detectCtx:mp.Queue, weights, source, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, classes=None, agnostic_nms=False, augment=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False, save_txt=False):
        self.detectInfoCtx = detectCtx
        self.weights = weights
        self.source = str(source)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.trace = no_trace
        self.save_txt = save_txt

        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        set_logging()
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.img_size, s=self.stride)

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        self.model.half()

        self.vid_path, self.vid_writer = None, None

        if self.view_img:
            self.view_img = check_imshow()
            cudnn.benchmark = True

        self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride)

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

        self.path = None
        self.img, self.im0s = None, None
        self.vid_cap = None

        self.t1 = self.t2 = self.t3 = 0.0

        self.max_conf = -1
        self.max_xyxy = None
        self.detectFlag = False

        self.sequentialHits = 0
        
    def run(self):       
        hitsCount = 0
        for self.path, self.img, self.im0s, self.vid_cap in self.dataset:
            self.img = torch.from_numpy(self.img).to(self.device)
            self.img = self.img.half()
            self.img /= 255.0
            if self.img.ndimension() == 3:
                self.img = self.img.unsqueeze(0)

            self.t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(self.img, augment=self.augment)[0]
            self.t2 = time_synchronized()

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            self.t3 = time_synchronized()
            
            for i, det in enumerate(pred):
                max_conf = -1
                max_xyxy = None
                detectFlag = False
                
                p, s, im0, frame = self.path[i], '%g: ' % i, self.im0s[i].copy(), self.dataset.count

                p = Path(p)

                if len(det):
                    detectFlag = True
                    
                    det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "          
                    
                    for *xyxy, conf, cls in reversed(det):
                        if conf > max_conf:
                            max_conf = conf
                            max_xyxy = xyxy
                        
                        if self.view_img:
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(max_xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)
                    
            hitsCount = hitsCount + 1 if detectFlag else 0
            if max_xyxy is not None:
                max_xyxy = [coord.cpu() for coord in max_xyxy if isinstance(coord, torch.Tensor)]
                
            self.detectInfoCtx.put((detectFlag, hitsCount, max_xyxy))
            self.print_detection_info(s)

    def print_detection_info(self, s):
        fps = 1E3 / (1E3 * (self.t2 - self.t1) + 1E3 * (self.t3 - self.t2))
        print(f'{s}Done. ({self.t2 - self.t1:.3f}s) Inference, ({self.t3 - self.t2:.3f}s) NMS, ({fps:.1f})FPS', flush=True)

class gimbalCtrl():
    def __init__(self, xyxy:mp.Queue, pid_info:mp.Queue, angleQueue:mp.Queue):
        self.xyxy = xyxy
        self.pid_info = pid_info
        self.angle = angleQueue
        self.pid = self._pid()
        self.yaw = self._motor(1, 0.0, 90)
        self.pitch = self._motor(2, 0.0, 45)
        
    def run(self):
        while True:
            xyxy_stat = self.xyxy.full()
            if xyxy_stat:
                pidErr = self.pid.pid_run(self.xyxy.get())
                m_flag1, m_flag2 = False, False
                
                if abs(pidErr[0]) != 0:
                    self.yaw.incrementTurnVal(int(pidErr[0]*100))
                else:
                    m_flag1 = True

                if abs(pidErr[1]) != 0:
                    self.pitch.incrementTurnVal(int(pidErr[1]*100))
                else:
                    m_flag2 = True

            pidInfoStat = self.pid_info.full()
            if pidInfoStat:
                self.pid_info.get()
                if xyxy_stat.empty():
                    self.pid_info.put("None")
                else:
                    self.pid_info.put(m_flag1 and m_flag2)
            
    def AngleLoop(self):
        while True:
            queueStat = self.angle.full()
            if queueStat:
                self.angle.get()
            self.angle.put((self.yaw.getAngle(), self.pitch.getAngle()))
            
    @staticmethod
    def _pid():
        from pid.pid import PID_Ctrl
        return PID_Ctrl()
    
    @staticmethod
    def _motor(ID, posInit, maxAngle):
        from pid.motor import motorCtrl
        return motorCtrl(ID, posInit, maxAngle)

def task(detectCtx:mp.Queue, xyxyQueue:mp.Queue):
    global angleQueue, centerQueue  # Add this line
    while True:
        if detectCtx.full():
            detectFlag, hitsCount, xyxy = detectCtx.get()
            
            if detectFlag and hitsCount > 3:
                pub_img['first_detect'] = True
                pub_bbox["x0"], pub_bbox['y0'], pub_bbox['x1'], pub_bbox["y1"] = xyxy
                
            manage_queue(xyxyQueue, xyxy)
            
            if angleQueue.full():
                pub_img['motor_yaw'], pub_img['motor_pitch'] = angleQueue.get()

            if centerQueue.full():
                stuts = centerQueue.get()
                if stuts != "None":
                    pub_img['camera_center'] = stuts

def gimbal(pQueue, PID_infoQueue, angleQueue):
    gCtrl = gimbalCtrl(pQueue, PID_infoQueue, angleQueue)
    gCtrl.run()

def main():
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        global xyxyQueue, centerQueue, angleQueue
        global mp_gimbal, mp_task
        xyxyQueue = mp.Queue(maxsize=1)
        centerQueue = mp.Queue(maxsize=1)
        angleQueue = mp.Queue(maxsize=1)
        mp_gimbal = mp.Process(target=gimbal, args=(xyxyQueue, centerQueue, angleQueue))
        mp_gimbal.start()
        
        detectInfoCtx = mp.Queue(maxsize=1) # detectFlag, hitsCount, max_xyxy
        mp_task = mp.Process(target=task, args=(detectInfoCtx, xyxyQueue))
        mp_task.start()                 
                
        weights = 'yolov7.pt'
        source = 'rtsp://127.0.0.2:8080/test'
        img_size = 640
        conf_thres = 0.25
        iou_thres = 0.45
        device = '0'
        view_img = True
        classes = None
        agnostic_nms = False
        augment = False
        project = 'runs/detect'
        name = 'exp'
        exist_ok = False
        no_trace = False
        save_txt = False
                
        detector = ObjectDetector(detectInfoCtx, weights, source, img_size, conf_thres, iou_thres, device, view_img, classes, agnostic_nms, augment, project, name, exist_ok, no_trace, save_txt)
        with torch.no_grad():
            detector.run()
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, stopping processes...")
        mp_gimbal.terminate()
        mp_task.terminate()
        
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
