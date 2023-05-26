import argparse
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size,non_max_suppression,scale_coords
from utils.torch_utils import select_device,  TracedModel
import threading
import numpy as np
import time


class model_person:
    def __init__(self,lane):  
        modelpath = 'my_models\\person_2023.pt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=f'{modelpath}', help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        weights,   imgsz, trace =  opt.weights,  opt.img_size, not opt.no_trace

        # Initialize
        device = select_device(opt.device)
        half = device.type == 'cpu'  # half precision only supported on CUDA

        # Load model
        print(f"{lane}號車道__人人人__模型載入")
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())           # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16
        
        model = model
        device =device
        half = half

        self.dperson = personsee(model,device,half,opt )


class personsee:
    def __init__(self,model ,device ,half ,opt  ):  
        self.opt = opt
        self.model = model
        self.device= device
        self.half= half
    def detect(self,img ,person_ylimittop ,person_ylimitdown ,person_xlimit ):
        person_ylimittop = person_ylimittop
        person_ylimitdown = person_ylimitdown
        person_xlimit = person_xlimit
        frame = img
        imgsz = 640
        stride = 32
        model = self.model
        device  =self.device
        half =self.half
        
        torch.set_num_threads(1)
        im0 = frame.copy()
        img = LoadImages(frame, img_size=imgsz, stride=stride).gogo()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # 如果 裝置有cuda 就帶入
        if device.type != 'cpu':
            # print(device.type)      # 確認是不是cuda
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

        # Apply NMS  避免同一物體被重複偵測  ，  具有最高分數的檢測框作為代表，然後排除其他檢測框。
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        

        
        # ===========================     Process detections

        head_cfg =   0
        person_cfg = 0
        helmet_cfg = 0

        for i, det in enumerate(pred):               
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()    # Rescale boxes from img_size to im0 size
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    label = f'{names[int(cls)]}'
                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int(xyxy[2])-int(xyxy[0])
                    h = int(xyxy[3])-int(xyxy[1])

                   

                    if label=="person" and x<person_xlimit and y>person_ylimittop and y<person_ylimitdown :
                        person_cfg = round(float(conf), 2)
                        print(f"{y} person   {person_cfg}")
                    if label=="head":
                        head_cfg = round(float(conf), 2)
                        print(f"{y} headd   {head_cfg}")
                    if label=="helmet":
                        helmet_cfg = round(float(conf), 2)
                        print(f"{y} helmet   {helmet_cfg}")
                        
        return person_cfg , helmet_cfg, head_cfg , im0
            


class d_person(model_person):
    def __init__(self, dt ,lane, frame = None , person_img=None,helmet_img=None,head_img=None):  
        super().__init__(lane)
        self.lock = threading.Lock()  # 創建一個鎖
        self.lane = lane
        self.frame = frame               # 更新影像    

        self.person_img = person_img
        self.helmet_img = helmet_img
        self.head_img = head_img


        # 初始
        self.initial_lane = lane
        self.initial_frame = frame

        self.initial_person_img = person_img
        self.initial_helmet_img = person_img
        self.initial_head_img = head_img

        
        self.dt = dt                     # 偵測頻率 2秒
        self.createTimer()

    def reset(self):
        self.lane = self.initial_lane            
        self.frame = self.initial_frame         # 更新影像


        self.person_img = self.initial_person_img
        self.helmet_img = self.initial_helmet_img
        self.head_img = self.initial_head_img


    def createTimer(self):
        t = threading.Timer(self.dt, self.detect)
        t.start()
    
    def detect(self):
        with self.lock:  # 加入鎖以保護這個方法
            if isinstance(self.frame, np.ndarray) and not np.array_equal(self.frame, self.initial_frame):
                if np.array_equal(self.helmet_img, self.initial_helmet_img) and np.array_equal(self.head_img, self.initial_head_img):
                    # 獲取影像的尺寸
                    height, width, channels = self.frame.shape
                    person_ylimittop = int(height)*0.1
                    person_ylimitdown = int(height)*0.8
                    person_xlimit = int(width)*0.9                    
                    p_cfg , ht_cfg , hd_cfg , d_img = 0,0,0,None
                    p_cfg , ht_cfg , hd_cfg , d_img= self.dperson.detect(self.frame ,person_ylimittop ,person_ylimitdown ,person_xlimit )

                
                    if p_cfg != 0:
                        self.person_img = d_img

                    if hd_cfg != 0 and not np.array_equal(self.person_img, self.initial_person_img):
                        self.head_img = d_img
                    if ht_cfg != 0 and not np.array_equal(self.person_img, self.initial_person_img):
                        self.helmet_img = d_img
                        
        self.createTimer()
        


# with torch.no_grad() 
# 是 PyTorch 中的上下文管理器 (context manager)，用於控制計算圖 (computation graph) 的計算過程是否需要進行梯度計算，以減少記憶體的使用，提高運行速度。
# 當進行模型的評估或預測時，通常不需要計算梯度，因此可以使用 with torch.no_grad() 上下文管理器來關閉 PyTorch 張量的自動求梯度功能，從而節省內存和提高運行速度。


# non_max_suppression() 
# 是一種物件偵測中的後處理技術，通常用於減少多餘的檢測框，以避免同一物體被重複偵測。具體而言，當物件偵測器產生多個重疊的檢測框時，non_max_suppression() 
# 會選擇一個具有最高分數的檢測框作為代表，然後排除其他檢測框。這樣做可以減少冗餘的檢測框，從而提高物件偵測的精確度和效率。

if __name__ == '__main__':
    # img = cv2.imread('person_image\\3_20230112150602.jpg')
    dt = 1
    lane = 1
    _dperson = d_person(dt,lane)  # 偵測


    import cv2

    # 打开视频文件
    video_path = r'D:\mike\newstyle\jojo20\bbb\Lane1.mp4'
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()

    # 循环读取视频帧
    while True:
        # 逐帧读取视频
        ret, frame = cap.read()

        _dperson.frame = frame          # 帶入影像
        print(_dperson.person_img)

        # 如果视频读取完毕，退出循环
        if not ret:
            break

        # 显示当前帧
        cv2.imshow('Frame', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频文件和关闭窗口
    cap.release()
    cv2.destroyAllWindows()
            

