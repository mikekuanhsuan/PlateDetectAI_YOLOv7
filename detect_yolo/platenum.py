import argparse
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel
import threading
import time

import numpy as np


class ModelNum:
    def __init__(self, lane):
        model_path = 'my_models/num_2023.pt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=f'{model_path}', help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        weights, imgsz, trace = opt.weights, opt.img_size, not opt.no_trace

        # Initialize
        device = select_device(opt.device)
        half = device.type == 'cpu'  # half precision only supported on CUDA

        # Load model
        print(f"{lane}號車道_車號_偵測模型載入")
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16

        self.dtruck = Trucksee(model, device, half, opt)


class Trucksee:
    def __init__(self, model, device, half, opt):
        self.opt = opt
        self.model = model
        self.device = device
        self.half = half

    def bubblesort(self, data, license_idx, conf_idx):
        n = len(data)
        for i in range(n-2):
            for j in range(n-i-1):
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
                    license_idx[j], license_idx[j+1] = license_idx[j+1], license_idx[j]
        for i in range(n-2):
            for j in range(n-i-1):
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
                    license_idx[j], license_idx[j+1] = license_idx[j+1], license_idx[j]

        delete_idx = []
        for i in range(n-2):
            for j in range(n-i-1):
                if data[j] == data[j+1]:
                    if conf_idx[j] > conf_idx[j+1]:
                        delete_idx.append(int(j+1))

                    if conf_idx[j+1] > conf_idx[j]:
                        delete_idx.append(int(j))
                if abs(int(data[j])-int(data[j+1])) < 5:
                    if conf_idx[j] > conf_idx[j+1]:
                        delete_idx.append(int(j+1))

                    if conf_idx[j+1] > conf_idx[j]:
                        delete_idx.append(int(j))

        my_list = list(set(delete_idx))
        my_list.sort(reverse=True)

        for i in my_list:
            del data[i]
            del license_idx[i]
            del conf_idx[i]

    def detect_license(self, img):
        imgsz = 640
        stride = 32
        model = self.model
        device = self.device
        half = self.half
        frame = img

        torch.set_num_threads(1)
        im0 = frame.copy()

        img = np.uint8(im0)
        imgr = img[:, :, 0]
        imgg = img[:, :, 1]
        imgb = img[:, :, 2]

        claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 18))
        claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 18))
        claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(10, 18))
        cllr = claher.apply(imgr)
        cllg = claheg.apply(imgg)
        cllb = claheb.apply(imgb)

        rgb_img = np.dstack((cllr, cllg, cllb))
        frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        img = LoadImages(frame, img_size=imgsz, stride=stride).gogo()
        names = model.module.names if hasattr(model, 'module') else model.names

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        old_img_w = old_img_h = imgsz
        old_img_b = 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=self.opt.augment)[0]

        with torch.no_grad():
            pred = model(img, augment=self.opt.augment)[0]

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        index_box = []
        license_idx = []
        conf_idx = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    label = f'{names[int(cls)]}'
                    conf_idx.append(float(conf))
                    index_box.append(int(xyxy[0]))
                    license_idx.append(label)

        if len(index_box) > 0:
            self.bubblesort(index_box, license_idx, conf_idx)
            license_idx = "".join(license_idx)
            return license_idx
        else:
            return "no_num"


class DPlateNum(ModelNum):
    def __init__(self, dt, lane, frame=None, predict=None, cnum=0):
        super().__init__(lane)
        self.lock = threading.Lock()
        self.lane = lane
        self.frame = frame
        self.predict = predict
        self.dt = dt
        self.cnum = 0

        self.initial_lane = lane
        self.initial_frame = frame
        self.initial_predict = predict
        self.initial_cnum = cnum

        self.create_timer()

    def reset(self):
        self.lane = self.initial_lane
        self.frame = self.initial_frame
        self.predict = self.initial_predict
        self.cnum = self.initial_cnum

    def reset_predict(self):
        self.predict = self.initial_predict

    def create_timer(self):
        t = threading.Timer(self.dt, self.detect)
        t.start()

    def get_predict(self):
        return self.predict

    def detect(self):
        with self.lock:
            if isinstance(self.frame, np.ndarray) and not np.array_equal(self.frame, self.initial_frame):
                if self.predict == self.initial_predict and self.cnum < 20:
                    self.predict = self.dtruck.detect_license(self.frame)
                    if self.predict == "no_num":
                        self.cnum += 1

                if self.predict == "no_num" and self.cnum < 20:
                    self.predict = self.dtruck.detect_license(self.frame)
                    if self.predict == "no_num":
                        self.cnum += 1

        self.create_timer()


if __name__ == '__main__':
    img = cv2.imread('bbb\\20230522145200_fplate.jpg')
    dt = 1
    lane = 1
    _dperson = DPlateNum( dt, lane, frame=None, predict=None, cnum=0)
    while True:
        _dperson.frame = img
        print(_dperson.predict)