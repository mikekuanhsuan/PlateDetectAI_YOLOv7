import argparse
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel
import threading
import numpy as np

class ModelTruck:
    def __init__(self, lane, model_path):
        self.model_path = model_path
        self.lane = lane
        self.d_truck = TruckSee(lane, model_path)

    def detect_truck(self, img):
        return self.d_truck.detect_truck(img)

class TruckSee:
    def __init__(self, lane, model_path):
        self.lane = lane
        self.opt = self.parse_arguments(model_path)
        self.model, self.device, self.half = self.initialize_model()

    @staticmethod
    def parse_arguments(model_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=model_path, help='model.pt 路徑')
        parser.add_argument('--img-size', type=int, default=640, help='推論尺寸 (像素)')
        parser.add_argument('--conf-thres', type=float, default=0.7, help='物體置信度閾值')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS 的 IOU 閾值')
        parser.add_argument('--device', default='', help='cuda 設備，例如 0 或 0,1,2,3 或 cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='篩選類別: --class 0，或 --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='無類別 NMS')
        parser.add_argument('--augment', action='store_true', help='增強推論')
        parser.add_argument('--no-trace', action='store_true', help='不追蹤模型')
        return parser.parse_args()

    def initialize_model(self):
        device = select_device(self.opt.device)
        half = device.type == 'cpu'  # 只有 CUDA 支援半精度

        # 載入模型
        print(f"{self.lane}號車道_偵測車輛模型載入")
        model = attempt_load(self.opt.weights, map_location=device)  # 載入 FP32 模型
        stride = int(model.stride.max())  # 模型步長
        imgsz = check_img_size(self.opt.img_size, s=stride)  # 檢查圖像尺寸
        if not self.opt.no_trace:
            model = TracedModel(model, device, self.opt.img_size)
        if half:
            model.half()  # 轉換為 FP16

        return model, device, half

    def detect_truck(self, img):
        frame = img
        imgsz = self.opt.img_size
        stride = 32

        torch.set_num_threads(1)
        im0 = frame.copy()
        img = LoadImages(frame, img_size=imgsz, stride=stride).gogo()

        # 取得類別名稱
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # 將圖像移至指定設備
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 轉換為 fp16/32
        img /= 255.0  # 0 - 255 轉換為 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 預熱
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # 執行一次

        # 禁用梯度計算
        with torch.no_grad():
            pred = self.model(img, augment=self.opt.augment)[0]

        # 應用 NMS 以避免重複偵測
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        predict_label = "no_truck"
        predict_img = []

        # 處理偵測結果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 將方框尺寸重新縮放為原圖大小
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}'

                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int(xyxy[2]) - int(xyxy[0])
                    h = int(xyxy[3]) - int(xyxy[1])
                    
                    if label == "Ftruck" or label == "Btruck":
                        predict_label = label
                        predict_img = im0

        return predict_label, predict_img

class DTruck(ModelTruck):
    def __init__(self, dt, lane, frame=None, predict=None, d_img=[]):
        model_path = 'my_models\\car_1209.pt'  # 指定模型路徑
        super().__init__(lane, model_path)
        self.lock = threading.Lock()  # 創建鎖
        self.lane = lane
        self.frame = frame  # 更新圖像
        self.predict = predict  # 辨識結果
        self.d_img = d_img  # 記錄模型識別的圖像

        self.initial_lane = lane
        self.initial_frame = frame
        self.initial_predict = predict
        self.initial_d_img = d_img

        self.dt = dt  # 偵測頻率：2 秒
        self.create_timer()

    def reset(self):
        self.lane = self.initial_lane
        self.frame = self.initial_frame  # 更新圖像
        self.predict = self.initial_predict  # 辨識結果
        self.d_img = self.initial_d_img

    def create_timer(self):
        t = threading.Timer(self.dt, self._go)
        t.start()

    def get_predict(self):
        return self.predict

    def _go(self):
        with self.lock:  # 添加鎖以保護此方法
            if self.frame is not None and isinstance(self.frame, np.ndarray):
                self.predict, self.d_img = self.detect_truck(self.frame)

        self.create_timer()


if __name__ == '__main__':
    import cv2
    import time

    # 創建一個 d_truck 類別的實例
    dt = 2  # 偵測頻率（秒）
    lane = 1  # 車道號碼
    frame = None  # 初始影格
    predict = None  # 初始預測結果
    d_img = []  # 初始偵測影像

    truck_detector = DTruck(dt, lane, frame, predict, d_img)

    # 開啟影片檔案
    video_path = 'D:/mike/newstyle/jojo20/bbb/car22.mp4'
    video_capture = cv2.VideoCapture(video_path)

    # 從影片檔案讀取影格，直到影片結束
    while video_capture.isOpened():
        # 讀取下一個影格
        ret, frame = video_capture.read()

        if ret:
            # 更新 truck_detector 物件的影格屬性
            truck_detector.frame = frame

            # 取得最新的預測結果
            prediction = truck_detector.get_predict()

            # 根據預測結果執行動作
            if prediction == 'Ftruck':
                print('偵測到__前方__卡車！')
                # 做一些動作

                # 儲存照片
                # current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # save_path = f"truck_{current_time}.jpg"
                # cv2.imwrite(save_path, frame)

                # 做一些動作
            elif prediction == 'Btruck':
                print('偵測到後方卡車！')
                # 做其他動作
            else:
                print('未偵測到卡車。')

            # 顯示影格（可選）
            cv2.imshow('影格', frame)
            
            # 若按下 'q' 鍵則結束迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 釋放影片擷取物件，並關閉所有開啟的視窗
    video_capture.release()
    cv2.destroyAllWindows()

        



# with torch.no_grad() 
# 是 PyTorch 中的上下文管理器 (context manager)，用於控制計算圖 (computation graph) 的計算過程是否需要進行梯度計算，以減少記憶體的使用，提高運行速度。
# 當進行模型的評估或預測時，通常不需要計算梯度，因此可以使用 with torch.no_grad() 上下文管理器來關閉 PyTorch 張量的自動求梯度功能，從而節省內存和提高運行速度。


# non_max_suppression() 
# 是一種物件偵測中的後處理技術，通常用於減少多餘的檢測框，以避免同一物體被重複偵測。具體而言，當物件偵測器產生多個重疊的檢測框時，non_max_suppression() 
# 會選擇一個具有最高分數的檢測框作為代表，然後排除其他檢測框。這樣做可以減少冗餘的檢測框，從而提高物件偵測的精確度和效率。