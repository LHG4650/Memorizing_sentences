
from multiprocessing.spawn import old_main_modules
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

import pandas as pd
import sys
import os
import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np
import copy
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LHG_FN.JPY_IMG import ImgWindow

from LHG_FN.Yolo_train_helper import yaml_make, yolo_train_commend
from LHG_FN.porridge_fn import pred_2_input, Porridge_NN, pred_to_input, Porridge_dataset
from DW_AI.DW_detect import DW_detect
import shutil
import seaborn as sns
import random
import math

from porridge import Porridge, pred_to_input_RF_train

root_path = "C:/Users/gusrm/Desktop/porg/FBF8_IC_porridgevision"
yolo_root_path = root_path + "/DATA_SET_V3/yolo_train_data"

img_folder = yolo_root_path+ "/images"

yolo_model_path = yolo_root_path + "/yolo_result"+"/220805_1000.pt"
RF_model_path = r'C:\Users\gusrm\Desktop\porg\FBF8_IC_porridgevision\DATA_SET_V3\model_save/RF_now.pt'

data_path = r'C:\Users\gusrm\Desktop\porg\FBF8_IC_porridgevision\DATA_SET_V3\csv_data\RF_train_N_label/RF_data.csv'

ui_path = "C:/Users/gusrm/Desktop/porg/speed_label_img/gui_v1.ui"

save_csv_path = root_path+ "/DATA_SET_V3/csv_data/0805_V3_label.csv"


Porridge_AI = Porridge()
Porridge_AI.InitializeWeight(yolo_model_path,RF_model_path)


main_window_form_class = uic.loadUiType(ui_path)[0]



class MainWindow(QtWidgets.QMainWindow, main_window_form_class):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.input_data = pd.read_csv(data_path).set_index('img_name')
        

        self.now_target = ""
        self.before_target = ""
        self.target_idx = 0

        
        self.get_new_target()
        self.img_update(self.now_target)
        self.square =[[0,0],[0,0]]

    # def get_new_target(self):       #2          #라벨이 없는것들 +
    #     target_list = pd.read_csv(data_path).set_index('img_name').index.tolist()
    #     before_pred = pd.read_csv(data_path).set_index('img_name')
    #     dumi_target=target_list[self.target_idx]

    #     self.before_target = copy.copy(self.now_target)
    #     self.now_target = dumi_target
        
    #     txt = before_pred.loc[dumi_target]
    #     txt = 'pred = ' + str(before_pred.loc[dumi_target]['pred']) + '// answer = ' + str(before_pred.loc[dumi_target]['answer'])
    #     self.class_V.setText(txt)
        
    #     self.target_idx += 1
        
        
        
    
    
    def get_new_target(self):      # Ver1       새 타겟을 찾는 과정 목적은 NG로 해당하는것을 찾음 10번 이상 찾으면 OK를 줌
        # 처음볼것 / 라벨이 없는것, 두번째로 볼것 IMG = NG인것
        target_lsit = copy.copy(self.input_data)
        target_lsit = target_lsit[(target_lsit['img_label'] != 1) & (target_lsit['img_label'] != 0)].index.tolist()       #라벨이 없는것
        #print(target_lsit)
        target_is_ok = 1
        count = 0

        while target_is_ok:
            dumi_target = random.choice(target_lsit)
            mid = self.input_data[['midx','midy']].loc[dumi_target].tolist()
            RF_result = Porridge_AI.InspactionImage(img_folder + "/" + dumi_target,mid[0],mid[1])
            if RF_result == 0:
                target_is_ok = 0
                self.before_target = copy.copy(self.now_target)
                self.now_target = dumi_target
            else:
                count +=1
            
            if count > 20:
                print('NG찾기가 쉽지 않습니다')
                target_is_ok = 0
                self.before_target = copy.copy(self.now_target)
                self.now_target = dumi_target


    def img_update(self,target):
        img = cv2.imread(img_folder + "/" + target)
        origin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_img = QImage(origin_img.data, origin_img.shape[1], origin_img.shape[0], QImage.Format_RGB888)
        origin_img = QPixmap.fromImage(origin_img)
        self.origin_img.setPixmap(origin_img)

        yolo_img, pred = Porridge_AI.AI_detect(img)
        mid = self.input_data[['midx','midy']].loc[target].tolist()     #RF 욜로 데이터 생성
        a_data_dt = pred_to_input_RF_train(pred,mid,target).set_index('img_name')

        pt = a_data_dt[['spx1','spy1','spx2','spy2',
                              'cham1x1','cham1y1','cham1x2','cham1y2',
                              'cham2x1','cham2y1','cham2x2','cham2y2',
                              'midx','midy'                           ]].loc[target].tolist()
        
        self.ImageUpdatePrecess(pt,yolo_img)

        RF_result = Porridge_AI.InspactionImage(img_folder + "/" + target,mid[0],mid[1])
        
        if RF_result:
            txt = 'RF_result = OK'
            self.prepred = 1
        else:
            txt = 'RF_result = NG'
            self.prepred = 0

        self.dist_V.setText(txt)

        self.now_rf_dt = a_data_dt




    def keyPressEvent(self, a0):
        img_label_now =999
        if a0.key() == Qt.Key_A:    #1 정상이미지
            print(self.now_target + '정상')
            img_label_now = 1

        elif a0.key() == Qt.Key_D:  #0 불량이미지
            print(self.now_target + '불량')
            img_label_now = 0

        elif a0.key() == Qt.Key_S:
            print(self.now_target + 'Re yolo')
            img_label_now =2

        elif a0.key() == Qt.Key_Z:
            print(self.now_target + '이전으로')
            img_label_now =3


        if img_label_now == 0 or img_label_now ==1:     #0이나 1
            self.save_data(img_label_now)               #저장
            self.get_new_target()                       #새 타겟 선정
            self.img_update(self.now_target)            #이미지 업데이트

        elif img_label_now == 2:        #애매함 #욜로가 이상함
            old = img_folder + "/" + self.now_target
            new = root_path + "/DATA_SET_V3/test_view/0805_need_yolo_label"
            shutil.copy(old,new)
            self.get_new_target()                       #새 타겟 선정
            self.img_update(self.now_target)            #이미지 업데이트
        
        elif img_label_now == 3:    #이전으로
            self.now_target = copy.copy(self.before_target)     #이전이미지 되돌려오기
            self.img_update(self.now_target)        #이전이미지 업데이트



    def save_data(self, result):
        self.now_rf_dt['img_label'].loc[self.now_target] = result
        self.input_data.loc[self.now_target]=self.now_rf_dt.loc[self.now_target]
        #self.input_data.to_csv(data_path)
        self.train_rf_model()
        
        self.square[self.prepred][result] +=1
        txt_1 = '왼쪽 머신 / 위쪽 사람'
        txt_2 = str(self.square[0][0]) +" / "+str(self.square[0][1]) + "\n" + str(self.square[1][0]) +" / "+str(self.square[1][1])
        
        self.LR_V.setText(txt_1)
        self.remain.setText(txt_2)
        
        
        



    def train_rf_model(self):
        origin_data = pd.read_csv(data_path).dropna()

        trai_data, test_data = train_test_split(origin_data, test_size=0.2  )
        train_dataset = Porridge_dataset(trai_data.reset_index(drop=True))
        test_dataset = Porridge_dataset(test_data.reset_index(drop=True))
        RandomForest = RandomForestClassifier()
        RandomForest.fit(train_dataset.x, train_dataset.y)
        y_pred = RandomForest.predict(test_dataset.x)
        print(classification_report(test_dataset.y, y_pred))
        print(confusion_matrix(test_dataset.y, y_pred))
        
        torch.save(RandomForest,RF_model_path)

    def ImageUpdatePrecess(self,pt,yolo_img):
        
        a = pt[0:4]
        b = pt[4:8]
        c = pt[8:12]
        mid = pt[12:14]


        get_line = []   # 라인들
        for i in [a,b,c]:
            line_dIc = {}
            dist_list = []
            for x in [i[0],i[2]]:
                for y in [i[1],i[3]]:
                    if not x*y ==0:
                        dist_list.append(math.dist([x,y],mid))
                        line_dIc[math.dist([x,y],mid)]=[int(x),int(y)]
            if not x*y == 0:
                dist_list.sort(reverse=True)
                dist_list = dist_list[:2]
                get_line.append(line_dIc[dist_list[0]])
                get_line.append(line_dIc[dist_list[1]])

        mid = [int(mid[0]),int(mid[1])]

        for i in get_line:
            yolo_img = cv2.line(yolo_img,i,mid,[255,255,255],5)    #선긋고
        
        yolo_img = cv2.line(yolo_img,mid,mid,[0,0,0],7)            #점찍고

        view_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB) 
        view_img = QImage(view_img.data, view_img.shape[1], view_img.shape[0], QImage.Format_RGB888)
        view_img = QPixmap.fromImage(view_img)
        self.view_img.setPixmap(view_img)


    def save_recode(self):
        # self.data.to_csv(row_data)
        print('a')
        #return super().keyPressEvent(a0)

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())