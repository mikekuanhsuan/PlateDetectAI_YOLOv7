import threading
from datetime import datetime

from getsql import db
from sendAlarm import *

class OccGo:
    def __init__(self, lane):
        self._datetime = None
        self._wvalue = None
        self.car_in = 0
        self.stable = None
        self.connect230 = 0
        self.connection_time = None
        self.disconnection_time = None
        self.lane = lane
        self.tagname = self._get_tagname()
        self._create_timer()

    def _get_tagname(self):
        if self.lane == 1:
            return "KY-T1HIST.WT-8006_T"
        elif self.lane == 2:
            return "KY-T1HIST.WT-8026_T"
        elif self.lane == 3:
            return "KY-T1HIST.WT-8046_T"
        else:
            return ""

    def _create_timer(self):
        t = threading.Timer(2, self._gogo)
        t.start()

    def _gogo(self):
        sql = f"""
            SET NOCOUNT ON;
            DECLARE @TopN INT;
            SET @TopN = 1;
            SELECT TOP (@TopN) *
            FROM (
                SELECT DateTime, Value, vValue, StartDateTime
                FROM History
                WHERE History.TagName IN ('{self.tagname}')
                AND wwRetrievalMode = 'Cyclic'
                AND wwResolution = 1000
                AND wwQualityRule = 'Extended'
                AND wwVersion = 'Latest'
                AND Value IS NOT NULL
            ) temp
            ORDER BY DateTime DESC;
            SET NOCOUNT OFF;"""
        try:
            with db() as occdb:
                sql_data_230 = list(occdb.get_datatable(sql, 230))

            if len(sql_data_230) > 0:
                self.connection_time = datetime.now()
                self.connect230 = 1
                self._datetime, self._wvalue = sql_data_230[0][:2]
                if self._wvalue < 8000:
                    self.car_in = 0
                    self.stable = None
                if self._wvalue > 8000:
                    self.car_in = 1
                if self._wvalue > 13000:
                    self.stable = 1
                
           
        except:
            self.connect230 = 0
            server_name = '230'
            self.disconnection_time = datetime.now()
            duration = 60
            SQL = f"""INSERT INTO [Image_recognition].[dbo].[ConnectionLogs] ([ServerName], [ConnectionTime], [DisconnectionTime], [DurationInSeconds]) 
                    VALUES ('{server_name}', '{self.connection_time}', '{self.disconnection_time}', {duration})
                    """
            with db() as OCCDB:
                OCCDB.run_cmd(SQL, 46)

            finish_txt = (f"斷線時間  {self.disconnection_time}")
            send_mail('gx.kao@advanced-tek.com.tw',"230斷線",finish_txt)


        self._create_timer()


if __name__ == '__main__':
    # 创建OccGo对象，传入相应的lane参数
    occ = OccGo(lane=1)

    # 获取当前时间
    current_time = datetime.now()

    # 打印初始状态
    print("Initial state:")
    print("Datetime:", occ._datetime)
    print("Wvalue:", occ._wvalue)
    print("Car in:", occ.car_in)
    print("Car in count:", occ.car_in_count)
    print()

    # 模拟等待一段时间
    # ...

    # 打印更新后的状态
    print("Updated state:")
    print("Datetime:", occ._datetime)
    print("Wvalue:", occ._wvalue)
    print("Car in:", occ.car_in)
    print("Car in count:", occ.car_in_count)
