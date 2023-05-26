import cv2
import os
import numpy as np






def save_person_pic(self , frame):
    if isinstance(frame, np.ndarray) and self.person_s==self.initial_person_s:
        foldetpath = f"{self.savepath}\\{self.img_id}\\{self.img_id}_person.jpg"  
        cv2.imwrite(f'{foldetpath}', frame)
        self.person_s = 1

def save_head_pic(frame, img_id, savepath):
    if isinstance(frame, np.ndarray):
        img_naem = f"{img_id}_person.jpg"
        foldetpath = f"{savepath}\\{img_id}\\{img_naem}"  
        cv2.imwrite(f'{foldetpath}', frame)

def save_helmet_pic(frame, img_id, savepath):
    if isinstance(frame, np.ndarray):
        foldetpath = f"{savepath}\\{img_id}\\{img_id}_person.jpg"  
        cv2.imwrite(f'{foldetpath}', frame)
        

def save_license_pic(frame , img_id, savepath):
    if isinstance(frame, np.ndarray):
        foldetpath = f"{savepath}\\{img_id}\\{img_id}_fplate.jpg"  
        UU = os.path.exists(foldetpath) 
        if not UU:
            cv2.imwrite(f'{foldetpath}', frame)


def save_carout_pic(frame, img_id, savepath):
    if isinstance(frame, np.ndarray):
        img_naem = f"{img_id}_F2.jpg"
        foldetpath = f"{savepath}\\{img_id}\\{img_naem}"  
        UU = os.path.exists(foldetpath) 
        if not UU:
            cv2.imwrite(f'{foldetpath}', frame)

def save_carout_pic_b(frame, img_id, savepath):
    if isinstance(frame, np.ndarray):
        img_naem = f"{img_id}_B2.jpg"
        foldetpath = f"{savepath}\\{img_id}\\{img_naem}"  
        UU = os.path.exists(foldetpath) 
        if not UU:
            cv2.imwrite(f'{foldetpath}', frame)

def save_carin_pic(frame, img_id, savepath ):
    
    if isinstance(frame, np.ndarray):
        img_naem = f"{img_id}_F1.jpg"
        foldetpath = f"{savepath}\\{img_id}\\{img_naem}"  
        print(f"{foldetpath}")
        UU = os.path.exists(foldetpath) 
        if not UU:
            print("SSSSSSSSSSSSSSSSSS##########")
            cv2.imwrite(f'{foldetpath}', frame)

def save_carin_pic_b(frame , img_id, savepath):
    if isinstance(frame, np.ndarray):
        img_naem = f"{img_id}_B1.jpg"
        foldetpath = f"{savepath}\\{img_id}\\{img_naem}"  
        # print(f"{foldetpath}")
        UU = os.path.exists(foldetpath) 
        if not UU:
            cv2.imwrite(f'{foldetpath}', frame)