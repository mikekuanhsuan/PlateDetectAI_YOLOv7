import threading
import time

from getsql import db


def get_occ_data(lane):
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

if __name__ == '__main__':
    # 測試程式
    lane = 1  # 設定 lane 的值
    occ_data = get_occ_data(lane)
    # 等待一段時間，確保檢測執行
    time.sleep(5)
    # 印出檢測結果
    print(occ_data)
