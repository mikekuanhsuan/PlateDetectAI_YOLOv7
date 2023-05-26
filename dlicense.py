import argparse
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel
import threading
import numpy as np


class ModelLicense:
    def __init__(self, lane):
        model_path = 'my_models/license_1213.pt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=f'{model_path}', help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
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
        print(f"{lane}號車道_車牌偵測模型載入")
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16

        self.trucksee = Trucksee(model, device, half, opt)

    def detect_license(self, img):
        return self.trucksee.detect_license(img)


class Trucksee:
    def __init__(self, model, device, half, opt):
        self.opt = opt
        self.model = model
        self.device = device
        self.half = half

    def detect_license(self, img):
        frame = img
        imgsz = 640
        stride = 32
        model = self.model
        device = self.device
        half = self.half

        torch.set_num_threads(1)
        im0 = frame.copy()
        img = LoadImages(frame, img_size=imgsz, stride=stride).gogo()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # 如果裝置有cuda就帶入
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=self.opt.augment)[0]

        # 不需要計算梯度
        with torch.no_grad():
            pred = model(img, augment=self.opt.augment)[0]

        # Apply NMS 避免同一物體被重複偵測，具有最高分數的檢測框作為代表，然後排除其他檢測框。
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        # =========================== Process detections
        predict_label = "no_license"
        predict_img = []

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    # label = f'{names[int(cls)]}'
                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int(xyxy[2]) - int(xyxy[0])
                    h = int(xyxy[3]) - int(xyxy[1])
                    new_im0 = im0[y:y + h, x:x + w]
                    predict_img = new_im0
                    predict_label = "license"

        return predict_label, predict_img


class DLicense(ModelLicense):
    def __init__(self, dt, lane, frame=None, predict_img=None, label=None, cnum=0):
        super().__init__(lane)
        self.lock = threading.Lock()  # 創建一個鎖
        self.lane = lane
        self.frame = frame  # 更新影像
        self.predict_img = predict_img  # 每秒更新結果
        self.predict_label = label  # 每秒更新結果
        self.cnum = cnum

        self.dt = dt  # 偵測頻率 3秒
        self.initial_lane = lane
        self.initial_frame = frame
        self.initial_predict_label = label
        self.initial_predict_img = predict_img
        self.initial_cnum = cnum

        self.create_timer()

    def reset(self):
        self.lane = self.initial_lane
        self.frame = self.initial_frame  # 更新影像
        self.predict_img = self.initial_predict_img  # 辨識結果
        self.predict_label = self.initial_predict_label  # 辨識結果
        self.cnum = self.initial_cnum  # 辨識結果

    def create_timer(self):
        t = threading.Timer(self.dt, self.detect)
        t.start()

    def detect(self):
        with self.lock:  # 加入鎖以保護這個方法
            if isinstance(self.frame, np.ndarray) and not np.array_equal(self.frame, self.initial_frame):
                # 初始
                if self.predict_label == self.initial_predict_label or self.predict_label == "no_license":
                    self.predict_label, self.predict_img = self.detect_license(self.frame)
               
                # if self.predict_label == self.initial_predict_label and self.cnum < 20:
                #     # 帶入影像辨識
                #     self.predict_label, self.predict_img = self.detect_license(self.frame)
                #     if self.predict_label == "no_license":
                #         self.cnum += 1

                # if self.predict_label == "no_license" and self.cnum < 20:
                #     # 帶入影像辨識
                #     self.predict_label, self.predict_img = self.detect_license(self.frame)
                #     if self.predict_label == "no_license":
                #         self.cnum += 1

        self.create_timer()


import cv2

if __name__ == '__main__':
    # Open the video file
    video_path = 'D:/mike/newstyle/jojo20/bbb/car22.mp4'
    cap = cv2.VideoCapture(video_path)

    # Create the DLicense instance
    lane = 1
    predict_img = None
    predict_label = None
    cnum = 0
    license_detector = DLicense(3, lane, None, predict_img, predict_label, cnum)

    # Read frames from the video and pass them to the license detector
    while cap.isOpened():
        ret, frame = cap.read()
        print(license_detector.predict_label)
        if not ret:
            break

        with license_detector.lock:
            license_detector.frame = frame

    # Release the video capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()
