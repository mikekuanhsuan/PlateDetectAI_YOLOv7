

from getsql import db
import time
import os


def delet_data(img_id):
    SQL = f"""
            DELETE FROM [Image_recognition].[dbo].[Fty_Photo]
            WHERE ID='{img_id}' 
            """
    with db() as OCCDB:
        OCCDB.run_cmd(SQL, 46)
        
def sql_license(platenum , imgid):
    SQL = f"""
            UPDATE Fty_Photo SET LicenseID='{platenum}' WHERE ID='{imgid}' 
            """
    with db() as OCCDB:
        OCCDB.run_cmd(SQL, 46)

    return 1


def update_workid(timeout,img_id,work_id,unhat,person_s):
    

    SQL = ""
    if work_id == " ":
        SQL = f"""
        UPDATE Fty_Photo SET Time2='{timeout}',Unsafety_hat='{unhat}',Unsafety_person='{person_s}' , WORK_ID=NULL 
        WHERE ID='{img_id}' 
        """
    else:      
        SQL = f"""
        UPDATE Fty_Photo SET Time2='{timeout}',Unsafety_hat='{unhat}',Unsafety_person='{person_s}' , WORK_ID='{work_id}' 
        WHERE ID='{img_id}' 
        """

    with db() as OCCDB:
        OCCDB.run_cmd(SQL, 46)
        

def sql_carin( img_id , lane , carin_time):
 
    SQL = f""" 
    INSERT INTO Fty_Photo(ID,FactoryID,Lane,Rep,Time1) 
    Values ('{img_id}','KY-T1HIST','{lane}','','{carin_time}') 
    """
    with db() as OCCDB:
        OCCDB.run_cmd(SQL, 46)