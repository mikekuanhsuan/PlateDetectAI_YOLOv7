import threading
import time
from threading import Lock
import cv2
import pymssql
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from cctvget import Camera
import configparser


class Camera_go():
    def __init__(self, lane):
        self.lock = threading.Lock()

        self.lane = lane

        config = configparser.ConfigParser()
        config.read('cctv.ini')
        self.savepath = "D:\car"
        self.f_camera = None
        self.m_camera = None
        self.b_camera = None
        if lane ==1:
            # 讀取各個區段的數值
            self.lane1_f_camera = config.get('lane1', 'f_camera_160')
            self.lane1_m_camera = config.get('lane1', 'm_camera_103')
            self.lane1_b_camera = config.get('lane1', 'b_camera_252')
            self.f_camera = Camera("192.168.1.160" , f"{self.lane1_f_camera}")
            self.m_camera = Camera("192.168.1.103" , f"{self.lane1_m_camera}")            
            self.b_camera = Camera("192.168.1.252" , f"{self.lane1_b_camera}")
        
        if lane == 2:
            # 讀取各個區段的數值
            self.lane1_f_camera = config.get('lane2', 'f_camera_159')
            self.lane1_m_camera = config.get('lane2', 'm_camera_4')
            self.lane1_b_camera = config.get('lane2', 'b_camera_57')
            self.f_camera = Camera("192.168.1.159", self.lane1_f_camera)
            self.m_camera = Camera("192.168.1.4", self.lane1_m_camera)            
            self.b_camera = Camera("192.168.1.57", self.lane1_b_camera)

        if lane == 3:
            # 讀取各個區段的數值
            self.lane1_f_camera = config.get('lane3', 'f_camera_158')
            self.lane1_m_camera = config.get('lane3', 'm_camera_101')
            self.lane1_b_camera = config.get('lane3', 'b_camera_122')
            self.f_camera = Camera("192.168.1.158", self.lane1_f_camera)
            self.m_camera = Camera("192.168.1.101", self.lane1_m_camera)            
            self.b_camera = Camera("192.168.1.122", self.lane1_b_camera)

        self.create_timer()

    def create_timer(self):
        t = threading.Timer(1, self.see)
        t.start()


    def see(self):
        if self.lane ==1:
            if not self.m_camera.flag:
                self.m_camera = Camera("192.168.1.103" , f"{self.lane1_m_camera}")
            if not self.f_camera.flag:
                self.f_camera = Camera("192.168.1.103" , f"{self.lane1_f_camera}")
            if not self.b_camera.flag:
                self.b_camera = Camera("192.168.1.252" , f"{self.lane1_b_camera}")

        if self.lane ==2:
            if not self.m_camera.flag:
                self.m_camera = Camera("192.168.1.159" , f"{self.lane1_m_camera}")
            if not self.f_camera.flag:
                self.f_camera = Camera("192.168.1.4" , f"{self.lane1_f_camera}")
            if not self.b_camera.flag:
                self.b_camera = Camera("192.168.1.57" , f"{self.lane1_b_camera}")

        if self.lane ==3:
            if not self.m_camera.flag:
                self.m_camera = Camera("192.168.1.158" , f"{self.lane1_m_camera}")
            if not self.f_camera.flag:
                self.f_camera = Camera("192.168.1.101" , f"{self.lane1_f_camera}")
            if not self.b_camera.flag:
                self.b_camera = Camera("192.168.1.122" , f"{self.lane1_b_camera}")

        self.create_timer()

        
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