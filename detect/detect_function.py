import time, sys, signal
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.torch_utils import select_device
from dataclasses import dataclass, field
from typing import List, Optional, Any


def signal_handler(sig, frame):
    print('Signal detected, shutting down gracefully')
    sys.exit(0)


@dataclass
class YoloParameters:
    weights: str
    source: str
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = ''
    classes: Optional[List[int]] = None
    save_img: bool = True
    agnostic_nms: bool = False
    augment: bool = False

@dataclass
class YOLO_Dataset:
    path: str
    img: np.ndarray = field(default_factory=lambda: np.array([]))
    im0s: np.ndarray = field(default_factory=lambda: np.array([]))
    vid_cap: Optional[Any] = None

class YoloDetector:
    def __init__(self, weights, source, img_size, conf_thres, iou_thres, device, classes=None, save_img=True, agnostic_nms=False, augment=False):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.classes = classes
        self.save_img = save_img
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.model, self.stride, self.imgsz = self.load_model()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

    def load_model(self):
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        return model, stride, imgsz

    def load_data(self):
        webcam = self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        return dataset

    def predict(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time.time()
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]
        t2 = time.time()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t3 = time.time()

        return pred, t1, t2, t3

    def run(self):
        set_logging()
        self.dataset = self.load_data()

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

        for path, img, im0s, vid_cap in self.dataset:
            pred, t1, t2, t3 = self.predict(img)
            self.display_prediction(pred, t1, t2, t3)

    def display_prediction(self, pred, t1, t2, t3):
        s = ""
        for i, det in enumerate(pred):
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        inference_time = (t2 - t1) * 1E3 
        nms_time = (t3 - t2) * 1E3
        total_time = (t3 - t1)
        fps = 1 / total_time if total_time > 0 else float('inf')
        print(f'{s}Done. ({inference_time:.1f}ms) Inference, ({nms_time:.1f}ms) NMS, {fps:.2f} FPS')

def YOLO_parameter(weights="yolov7.pt", source='0', img_size=640, conf_thres=0.25, iou_thres=0.45,
                   device='', classes=None, save_img=True, agnostic_nms=False, augment=False) -> YoloParameters:
    return YoloParameters(weights, source, img_size, conf_thres, iou_thres, device, classes, save_img, agnostic_nms, augment)

def runDetection(para: YoloParameters):
    detector = YoloDetector(
        weights=para.weights,
        source=para.source,
        img_size=para.img_size,
        conf_thres=para.conf_thres,
        iou_thres=para.iou_thres,
        device=para.device,
        classes=para.classes,
        save_img=para.save_img,
        agnostic_nms=para.agnostic_nms,
        augment=para.augment
    )
    detector.run()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    rtspUrl = 'rtsp://127.0.0.' + str(np.random.randint(1, 256)) + ':8080/video_feed'
    yoloPara = YOLO_parameter(weights="20241127.pt", source=rtspUrl,img_size=640)
    runDetection(yoloPara)

if __name__ == '__main__':
    main()
