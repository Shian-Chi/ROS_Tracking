import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.torch_utils import select_device
from dataclasses import dataclass


@dataclass
class YoloParameters:
    weights: str
    source: str
    img_size: int
    conf_thres: float
    iou_thres: float
    device: str
    classes: None
    agnostic_nms: bool
    augment: bool
    

class YoloDetector:
    def __init__(self, weights, source, img_size, conf_thres, iou_thres, device, classes=None, agnostic_nms=False, augment=False):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.classes = classes
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
            dataset = LoadStreams(self.source, self.img_size, self.stride)
        else:
            dataset = LoadImages(self.source, self.img_size, self.stride)
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
        dataset = self.load_data()

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

        for _, img, _, _ in dataset:
            pred, t1, t2, t3 = self.predict(img)
            self.display_prediction(pred, t1, t2, t3)

    def display_prediction(self, pred, t1, t2, t3):
        s = ""
        for i, det in enumerate(pred):
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        inferenceTime = 1E3 * (t2 - t1)
        NMS_Time = (1E3 * (t3 - t2))
        print(f'{s}Done. ({inferenceTime:.1f}ms) Inference, ({NMS_Time:.1f}ms) NMS, {(1E3/inferenceTime+NMS_Time)}FPS')


def YOLO_parameter() -> YoloParameters:
    # YOLO Settings
    weights = 'landpad20240522.pt'
    source = 'rtsp://127.0.0.' + str(np.random.randint(1, 256)) + ':8080/video_feed'
    img_size = 480
    conf_thres = 0.3
    iou_thres = 0.45
    device = '0'
    classes = None
    agnostic_nms = False
    augment = False
    return YoloParameters(weights, source, img_size, conf_thres, iou_thres, device, classes, agnostic_nms, augment)


def runDetection(para: YoloParameters):
    detector = YoloDetector(
        para.weights,
        para.source,
        para.img_size,
        para.conf_thres,
        para.iou_thres,
        para.device,
        para.classes,
        para.agnostic_nms,
        para.augment
    )
    detector.run()


def main():
    yoloPara = YOLO_parameter()
    runDetection(yoloPara)


if __name__ == '__main__':
    main()
