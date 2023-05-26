import os
import configparser
import pymssql

class db():
    def __init__(self):
        proDir = os.getcwd()
        configPath = os.path.join(proDir, "sql_information.ini")
        config = configparser.ConfigParser()
        config.read(configPath)

        SQLpath = config['SQL']

        self.server_46 = SQLpath['server_46']
        self.user_46 = SQLpath['user_46']
        self.password_46 = SQLpath['password_46']
        self.database_46 = SQLpath['database_46']

        self.server_47 = SQLpath['server_47']
        self.user_47 = SQLpath['user_47']
        self.password_47 = SQLpath['password_47']
        self.database_47 = SQLpath['database_47']

        self.server_230 = SQLpath['server_230']
        self.user_230 = SQLpath['user_230']
        self.password_230 = SQLpath['password_230']
        self.database_230 = SQLpath['database_230']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    

    def conn(self, D):
        if D == 46:
            self.connect = pymssql.connect(server=self.server_46, user=self.user_46 , password=self.password_46 , database=self.database_46)  
        if D == 230:
            self.connect = pymssql.connect(server=self.server_230 , user=self.user_230 ,  password=self.password_230 , database=self.database_230 )
        if D == 47:
            self.connect = pymssql.connect(server=self.server_47 , user=self.user_47 ,  password=self.password_47 , database=self.database_47 )
        
        self.cursor = self.connect.cursor()



    def get_datatable(self, sql, D):
        self.conn(D)
        self.cursor.execute(sql)
        x = self.cursor.fetchall()
        self.close()
        return x

    def run_cmd(self, sql, D):
        self.conn(D)
        if hasattr(self, 'cursor'):
            self.cursor.execute(sql)
        if hasattr(self, 'connect'):
            self.connect.commit()
        self.close()

    def close(self):
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'connect'):
            self.connect.close()




if __name__ == '__main__':
    OCCDB = db()
    SQL = f"SELECT TOP (1000) [WORK_ID],[CAR_ID],[SCALE_NO] FROM [ZDB].[dbo].[OCC_Despatch] WHERE DEPT_ID='KY-T1HIST' AND O_DATE='2022-12-09' AND SCALE_NO='2'"
    # SQL = f"DELETE FROM Fty_Photo WHERE ID='{img_ID}'"
    # 得到230 TABLE
    x= OCCDB.get_datatable(SQL,230)
    x = list(x)
    print(len(x))
    if len(x)>1:
        for i in x :
            print(i[0])

