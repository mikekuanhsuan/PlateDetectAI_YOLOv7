import threading
import time
from threading import Lock
import cv2
import pymssql
from datetime import datetime
from sendAlarm import *

class Camera:
    def __init__(self, lane, rtsp_link):
        self.lane = lane
        self.rtsp_link = rtsp_link

        self.flag = True
        self.last_frame = None
        self.last_ready = None
        self.lock = Lock()

        self.failure_count = 0  # 失敗次數計數器
        self.max_failure_count = 5  # 指定的最大失敗次數

        self.failure_saved = False  # 標記錯誤資訊是否已儲存到資料庫

        capture = cv2.VideoCapture(self.rtsp_link)
        self.fps = capture.get(cv2.CAP_PROP_FPS) % 100
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(capture,), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def get_ready(self):
        return self.last_ready

    def save_failure_to_database(self, camera_id, error_message):
        # 建立資料庫連接
        conn = pymssql.connect(server='192.168.101.46\SQLEXPRESS', user='sa', password='123456', database='Image_recognition')

        # 取得目前的時間戳記
        timestamp = datetime.now()

        # 插入失敗紀錄到資料庫表中
        cursor = conn.cursor()
        print("dfsf")
        print(camera_id, timestamp, error_message)
        cursor.execute("INSERT INTO CameraFailureRecord (camera_id, timestamp, error_message) VALUES (%s, %s, %s)",
                       (camera_id, timestamp, error_message))
        conn.commit()

        # 關閉資料庫連接
        conn.close()

    def rtsp_cam_buffer(self, capture):
        while self.flag:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()

                if not self.last_ready:
                    self.failure_count += 1  # 增加失敗次數計數

                    if self.failure_count == self.max_failure_count and not self.failure_saved:
                        self.flag = False
                        print(f"{self.rtsp_link} 發生問題")  # 連結失敗，輸出錯誤訊息
                        finish_txt = f"連結失敗：{self.rtsp_link}"
                        send_mail('gx.kao@advanced-tek.com.tw',"相機連線失敗通知",finish_txt)

                        
                        self.save_failure_to_database(self.lane, f"連結失敗：{self.rtsp_link}")  # 儲存失敗紀錄到資料庫
                        self.failure_saved = True  # 將標記設置為已儲存

            self.fps = 10
            time.sleep(1 / self.fps)
        # print(f"{self.rtsp_link} 攝影機已關閉")

    def get_frame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None



if __name__ == '__main__':
    # 建立Camera物件，指定車道和RTSP連結
    lane = "192.168.1.160"
    rtsp_link = "rtsp://admin:123456@192.168.1.160/profixle1"
    camera = Camera(lane, rtsp_link)

    # 設定視窗名稱和大小
    window_name = "Camera Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # 調整視窗大小

    # 模擬測試，每隔1秒輸出相機準備狀態和取得影像幀
    while True:
        ready = camera.get_ready()
        frame = camera.get_frame()

        if ready:
            print("相機準備就緒")
        else:
            print("相機尚未準備")

        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        time.sleep(1)