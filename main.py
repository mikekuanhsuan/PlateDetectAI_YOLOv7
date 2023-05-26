import cv2
import os
import threading
import time
import configparser
import numpy as np
import datetime

from saveimg  import *
from getpbb import OccGo

from getsql import db
from getsqlin import *
from detect_yolo.truck import DTruck
from detect_yolo.dlicense import DLicense
from detect_yolo.platenum import DPlateNum
from detect_yolo.person import d_person

from runcctv import Camera_go

class Eyes:
    def __init__(self, lane):
        config = configparser.ConfigParser()
        config.read('savpath.ini')  # 取得 savepath 的值
        savepath = config.get('Settings', 'savepath')
        self.savepath = savepath   # 使用取得的值來設定 self.savepath

        self.happycctv = Camera_go(lane)
        self.occ_instance = OccGo(lane)        # 地磅重量

       # =====================================================
        # dt = 2  # 偵測頻率（秒）
        # self.dtruck = DTruck(dt, lane, frame=None, predict= None , d_img = [] )

        dt= 3  # 每3秒偵測
        self.ilicense = DLicense(dt, lane, frame=None, predict_img=None, label=None, cnum=0)

        # dt = 1
        self.iplatenum = DPlateNum( dt, lane, frame=None, predict=None, cnum=0)

        # dt= 1  # 每3秒偵測
        self.iperson = d_person( dt ,lane, frame = None , person_img=None,helmet_img=None,head_img=None)
        



class Girls(Eyes):
    def __init__(self, lane, factoryid="KY-T1HIST", img_id=0, work_id=" ", carin_time=0, stay_time=0, platenum=0, person_s=0, head_s=0, helmet_s=0):
        super().__init__(lane)
        self.img_id = img_id           # ID
        self.factoryid = factoryid
        self.lane = lane
        self.platenum = platenum       # 車號
        self.person_s = person_s
        self.head_s = head_s
        self.helmet_s = helmet_s
        self.stay_time = stay_time     # 滯留時間
        self.carin_time = carin_time   # 入磅秤時間
        self.work_id = work_id         # 工作ID
      
        self.license_sql = 0
        
     


        # 初始值
        self.initial_img_id = img_id
        self.initial_work_id = work_id
        self.initial_stay_time = stay_time 
        self.initial_carin_time = carin_time 
        self.initial_platenum = platenum 


        self.initial_person_s = person_s
        self.initial_head_s = head_s
        self.initial_helmet_s = helmet_s
        self.create_timer()
  

    def reset(self):
        self.img_id = self.initial_img_id     
        self.work_id = self.initial_work_id
        self.stay_time = self.initial_stay_time     
        self.carin_time = self.initial_carin_time  
        self.platenum = self.initial_platenum 

        self.person_s = self.initial_person_s
        self.head_s = self.initial_head_s
        self.helmet_s = self.initial_helmet_s
        self.license_sql = 0


    def create_timer(self):
        t = threading.Timer(1, self.go)
        t.start()

    def detect_plate(self):
        if self.happycctv.f_camera is not None and  self.happycctv.f_camera.get_frame() is not None:
            self.ilicense.frame =  self.happycctv.f_camera.get_frame()
  
  

    def get_occ_data(self, lane):
        sql_datas_occ = None
        time_date = time.strftime('%Y-%m-%d', time.localtime())
        # 最新一筆晨家資料
        sql = f"""
                SELECT TOP 1 [WORK_ID], [CAR_ID], [SCALE_NO], O_DATE
                FROM [ZDB].[dbo].[OCC_Despatch]
                WHERE DEPT_ID = 'KY-T1HIST' AND SCALE_NO = '{lane}' AND O_DATE = '{time_date}'
                ORDER BY [WORK_ID] DESC;
                """
        with db() as occ_db:
            sql_datas_occ = list(occ_db.get_datatable(sql, 230))
        return sql_datas_occ
    

    def no_id(self):
        ##### 入磅時間
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        self.carin_time= now_str # 入磅秤時間        

        ##### 建立ID
        now = datetime.datetime.now()
        self.img_id = now.strftime('%Y%m%d%H%M%S')

        #####  建立資料夾
        foldetpath = f"{self.savepath}\\{self.img_id}"   # FOLDER
        UU = os.path.exists(foldetpath)
        if not UU:
            os.makedirs(foldetpath)
        # 入磅------------------------SQL part 1
        sql_carin(self.img_id , self.lane , self.carin_time)
    
    def _id_person(self):
        if isinstance(self.happycctv.m_camera.get_frame(), np.ndarray):
            self.iperson.frame = self.happycctv.m_camera.get_frame()   
            
            if not np.array_equal(self.iperson.head_img, self.iperson.initial_head_img):
                if isinstance(self.iperson.head_img, np.ndarray) and self.head_s==self.initial_head_s:
                    save_head_pic(self.iperson.head_img , self.img_id, self.savepath)         # 儲存入磅影像
                    self.head_s = 1                              
            
            if not np.array_equal(self.iperson.helmet_img, self.iperson.initial_helmet_img):
                if isinstance(self.iperson.helmet_img, np.ndarray) and self.helmet_s==self.initial_helmet_s:
                    save_helmet_pic(self.iperson.helmet_img , self.img_id, self.savepath)         # 儲存入磅影像
                    self.helmet_s = 1

    def __stay_time(self):
        ##### 滯留時間(秒)
        if self.stay_time <300:
            self.stay_time += 1

    def ___update_workid(self):
        ##################  晨家資料
        sql_datas_occ = self.get_occ_data(self.lane)

        #################  
        if len(sql_datas_occ) > 0:
            for value in sql_datas_occ:
                workid = str(value[0]).strip()
                carid = str(value[1]).strip()
                odate = str(value[3]).strip()
                if  self.platenum == carid:
                    self.work_id = workid
        
        ##### 有沒有帽子
        unhat = 0
        if self.helmet_s ==0:
            unhat = 1

        ##### 有沒有人
        person_s = 0
        foldetpath = f"{self.savepath}\\{self.img_id}\\{self.img_id}_person.jpg"  
        UU = os.path.exists(foldetpath) 
        if UU:
            person_s=1

        ##### 更新work_id
        timeout = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        update_workid(timeout , self.img_id , self.work_id , unhat , person_s)

    def close_pdata(self):
        ################# 關閉攝影機
        self.happycctv.f_camera.flag = False
        self.happycctv.f_camera.last_frame = None
        self.happycctv.m_camera.flag = False
        self.happycctv.m_camera.last_frame = None
        self.happycctv.b_camera.flag = False
        self.happycctv.b_camera.last_frame = None

        # 清除資料
        # self.dtruck.reset()       # 車輛 >> 物件屬性初始化
        self.ilicense.reset()     # 車牌 >> 物件屬性初始化
        self.iplatenum.reset()    # 車號 >> 物件屬性初始化
        self.iperson.reset()    # 車號 >> 物件屬性初始化
        self.reset()              # sql重要屬性 >> 物件屬性初始化
    def go(self):
        self.detect_plate()

        carin = 0
        if self.occ_instance.connect230 ==1:
            carin = self.occ_instance.car_in
        else:
            if self.ilicense.predict_label!="no_license":
                carin = 1
            else:
                carin = 0
      
        # print(f"{self.lane} 車道       {carin}車    {self.img_id}   {self.stay_time}  等待時間    {self.occ_instance._wvalue}     {self.occ_instance.stable}")
        # 有車
        if carin == 1 :
            self.__stay_time()
            
            if self.img_id==0 and self.carin_time==0:
                self.no_id()
            
            if self.img_id!=0 and self.carin_time!=0:
                ##### 偵測人
                self._id_person()
               
            if self.occ_instance.stable==1:
                # 車輛
                save_carin_pic(self.happycctv.f_camera.get_frame() , self.img_id, self.savepath)        # 儲存入磅影像
                save_carin_pic_b(self.happycctv.b_camera.get_frame(), self.img_id, self.savepath)   

                # 車牌
                if self.ilicense.predict_label!="no_license":
                    if isinstance(self.ilicense.predict_img, np.ndarray):
                        save_license_pic(self.ilicense.predict_img , self.img_id, self.savepath)
                        self.iplatenum.frame =  self.ilicense.predict_img    #### 帶入車牌影像辨識車號

                # 車號
                if self.iplatenum.predict!=self.iplatenum.initial_predict  and self.iplatenum.predict!="no_num":
                    if self.license_sql ==0:
                        self.license_sql = sql_license(self.iplatenum.predict , self.img_id)
                        self.platenum = self.iplatenum.predict


        ######## 沒車
        if carin == 0:
            ############# 90
            if self.img_id!=0 and self.stay_time<90:
                delet_data(self.img_id)

            ############# 120
            if self.img_id!=0 and self.stay_time>120:
                ################# 打印 
                print(self.img_id,self.work_id,self.carin_time,self.platenum,self.platenum)

                ######### 前影像儲存
                save_carout_pic(self.happycctv.f_camera.get_frame(), self.img_id, self.savepath)
                save_carout_pic_b(self.happycctv.b_camera.get_frame(), self.img_id, self.savepath)

                ######### 比對工單
                self.___update_workid()
                
                ######## 關閉攝影機跟參數
                self.close_pdata()


            
                
        self.create_timer()


if __name__ == '__main__':
    
   

    admin_l1 = Girls(1, factoryid="KY-T1HIST")
    admin_l1 = Girls(2, factoryid="KY-T1HIST")
    admin_l1 = Girls(3, factoryid="KY-T1HIST")


