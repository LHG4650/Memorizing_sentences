#https://developer-mistive.tistory.com/59

from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QImage,QPixmap, QPalette, QColor, QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui


from DW_AI.DW_detect import DW_detect
#from DW_utils.utils.torch_utils import select_device
from LHG_FN.batch_detect_fn import fh2_img_const_zero_bler, set_log, write_log
from DW_utils.utils.torch_utils import select_device

import cv2
import time
import torch
import math
import sys
import winsound
import random
import os

try:
    os.chdir(sys._MEIPASS)
    print(sys._MEIPASS)
except:
    os.chdir(os.getcwd())

test_state = 0
origin_img = cv2.imread(r'C:\Users\gusrm\Desktop\porg\STF7_OD_-YOLO-batchdetect-Maker\dataset\STF7_OD_yolo\origin/22.08.04_18.49.40.png')



fh1_rtsp_url = "rtsp://admin:dhfh12!@@192.168.114.38:558/0/0/media.smp"   #rtsp://admin:dhfh12!@@192.168.114.38:558/3/0/media.smp //1로 정상 RTSP
fh1_yolo_model_path = "custom/2fh_yolo_model.pt"                          #"custom/1fh_yolo_model.pt"

fh2_rtsp_url = "rtsp://admin:dhfh12!@@192.168.114.38:558/0/0/media.smp"
fh2_yolo_model_path = "custom/2fh_yolo_model.pt"

ui_path = "custom/gui_v02.ui"

main_window_form_class = uic.loadUiType(ui_path)[0]    

class MainWindow(QtWidgets.QMainWindow, main_window_form_class):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.F1_position = [0,0]                                             #1번 영상 보정된 클릭값 저장
        self.F2_position = [0,0]                                             #2번 영상 보정된 클릭값
        self.F1_ready = 0                                                    #1번영상 L,R 좌표 되었는지 확인
        self.F2_ready = 0                                                    #2번영상 L,R 좌표 되었는지 확인
        self.F1_L_Position = [0,0]                                           #1번영상 L측 좌표저장
        self.F1_R_Position = [0,0]                                           #1번영상 R측 좌표저장    
        self.F2_L_Position = [0,0]                                           #2번영상 L측 좌표저장
        self.F2_R_Position = [0,0]                                           #2번영상 R측 좌표저장
        self.img_mul = 1
        self.dist_log_file_path = "not"

        self.Watcher1 = Fh_watcher(self,fh1_yolo_model_path,fh1_rtsp_url)
        self.Watcher1.start()
        self.Watcher1.ImageUpdate.connect(self.ImageUpdateSlot1)
        self.Watcher1.dist_num.connect(self.distUpdateSlot1)

        self.Watcher2 = Fh_watcher(self,fh2_yolo_model_path,fh2_rtsp_url)
        self.Watcher2.start()
        self.Watcher2.ImageUpdate.connect(self.ImageUpdateSlot2)
        self.Watcher2.dist_num.connect(self.distUpdateSlot2)

        self.Alarmer = Fh_alarm(self)
        self.Alarmer.start()
        #알람관련
        self.Alarmer.alarm_btn_signal.connect(self.alarm_btn_change)        #알람쓰레드에서 알람버튼 표시 변경할때 나오는 시그널
        self.Alarmer.alarm_background_signal.connect(self.Alarm_background_change)      #알람쓰레드에서 프로그램 배경색 변경할 때 나오는 시그널
        self.alarm_btn.clicked.connect(self.alarm_stop)                     #알람 정지버튼 클릭


        self.alarm_timer = QTimer(self)                             
        self.alarm_timer.start(1000)                                        # 1초마다
        self.alarm_timer.timeout.connect(self.alarm_Slot)   

        self.F1_L.clicked.connect(self.Save_position_1_L)                   #1로 L측 좌표 저장 버튼
        self.F1_R.clicked.connect(self.Save_position_1_R)                   #1로 R측 좌표 저장 버튼
        self.F2_L.clicked.connect(self.Save_position_2_L)                   #2로 L측 좌표 저장 버튼
        self.F2_R.clicked.connect(self.Save_position_2_R)                   #2로 R측 좌표 저장 버튼
        self.Img_mul_edit.setValidator(QDoubleValidator(0.5, 3, 2, self))   #이미지 배율 변경 제한 (0.5~3까지 소수점 2자리 까지 가능)
        self.Img_mul_btn.clicked.connect(self.Img_mul_change)               #이미지 배율 변경 클릭 및 반응

        self.timer = QTimer(self)                                           # timer 변수에 QTimer 할당
        self.timer.start(30*1000)                                             # 10000msec(10sec) 마다 반복
        self.timer.timeout.connect(self.log_save_Slot)                      # start time out시 연결할 함수 #로그는 10초에 한번씩 저장(현재기준)

    def Img_mul_change(self):                                               #입력받은 이미지 배율 수치 반영
        self.img_mul = float(self.Img_mul_edit.text())
        self.Watcher1.img_mul = self.img_mul
        self.Watcher2.img_mul = self.img_mul

    def alarm_stop(self):                                                   #알람 정지지상태 15분 유지됨 
        self.alarm_btn.setText("알람 정지2")
        self.Alarmer.Alarm_state = 2
        pal = QPalette()
        pal.setColor(QPalette.Background, QColor(240,240,240))              #배경색 정상화
        self.setPalette(pal)

    def alarm_Slot(self):                                                   #1초마다 알람쓰레드에 배치거리 전달
        self.Alarmer.f1_dist = self.Watcher1.near_dist
        self.Alarmer.f2_dist = self.Watcher2.near_dist

    def alarm_btn_change(self,data):
        self.alarm_btn.setText(data) 

    def Alarm_background_change(self, data):                                #배치 넘어올때 배경을 무작위 색상으로 전달  
        RR = random.randrange(0,255)
        GG = random.randrange(0,255)
        BB = random.randrange(0,255)
        if data == "0":                                                     #정상 판정시 RGB 240 (기본색상)
            RR = 240
            GG = 240
            BB = 240

        print("alarm 발생")
        pal = QPalette()
        pal.setColor(QPalette.Background, QColor(RR,GG,BB))
        self.setPalette(pal)

    def log_save_Slot(self):                                                #로그 기록 
        print('log_save_Slot 작동/' ,self.Watcher1.near_dist,'/',self.Watcher2.near_dist       )
        if self.dist_log_file_path == "not":
            self.dist_log_file_path = set_log('1_2')
        write_log(self.dist_log_file_path,self.Watcher1.near_dist,self.Watcher2.near_dist)  #self.Watcher2.near_dist)

    def ImageUpdateSlot1(self, Image):                               #이미지 이벤트 발생시 image 를 가져옴
        img = QPixmap.fromImage(Image)                              #Qpixmap 으로 이미지 변환되어
        self.f1_monitor.setPixmap(img)  

    def ImageUpdateSlot2(self, Image):                               #이미지 이벤트 발생시 image 를 가져옴
        img = QPixmap.fromImage(Image)                              #Qpixmap 으로 이미지 변환되어
        self.f2_monitor.setPixmap(img)  

    def distUpdateSlot1(self, data):                             #배치거리 150 이상이면 안보임으로 표출되게함
        if data > 149:
            txt_MMM = '배치거리 = 안보임'
        else:
            txt_MMM = '배치거리 = '+ str(round(data,2))
        self.f1_dist.setText(txt_MMM)

    def distUpdateSlot2(self, data):                             #배치거리 150 이상이면 안보임으로 표출되게함
        if data > 149:
            txt_MMM = '배치거리 = 안보임'
        else:
            txt_MMM = '배치거리 = '+ str(round(data,2))
        self.f2_dist.setText(txt_MMM)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:           #마우스 클릭 이벤트 (클릭하면 실행됨)

        x1 = a0.x()-self.f1_monitor.geometry().x()                     #x축 이미지 최소마진
        y1 = a0.y()-self.f1_monitor.geometry().y()
        
        #- a0.y() + self.f1_monitor.y() + self.f1_monitor.height() #y축은 높이 위에서 클릭지점을 가져옴/ 우측하단을 0 으로 만들고자 반전줌

        x2 = a0.x()-self.f2_monitor.geometry().x()
        y2 = a0.y()-self.f2_monitor.geometry().y()
        #y2 = - a0.y() + self.f2_monitor.y() + self.f2_monitor.height()

        """좌표의 원점은 이미지의 우측 하단을 0점으로 좌표변경된다 """

        self.F1_position = [int(x1/self.img_mul) ,int(y1/self.img_mul)]
        self.F2_position = [int(x2/self.img_mul) ,int(y2/self.img_mul)]
        
        M1_Txt = "마우스 좌표 : X={0}, Y={1}".format(x1,y1)
        M2_Txt = "마우스 좌표 : X={0}, Y={1}".format(x2,y2)

        self.f1_position.setText(M1_Txt)                          #UI에 좌표 표시
        self.f2_position.setText(M2_Txt)
        return super().mousePressEvent(a0)

    def Save_position_1_L(self):
        self.F1_L_Position = self.F1_position
        M_txt = "X={0}, Y={1}".format(self.F1_L_Position[0],self.F1_L_Position[1])      #bub1 클릭버튼 우측에 좌표 표출
        self.F1_L_V.setText(M_txt)
        self.check_save_point()

    def Save_position_1_R(self):
        self.F1_R_Position = self.F1_position
        M_txt = "X={0}, Y={1}".format(self.F1_R_Position[0],self.F1_R_Position[1])      #bub1 클릭버튼 우측에 좌표 표출
        self.F1_R_V.setText(M_txt)
        self.check_save_point()

    def Save_position_2_L(self):
        self.F2_L_Position = self.F2_position
        M_txt = "X={0}, Y={1}".format(self.F2_L_Position[0],self.F2_L_Position[1])      #bub1 클릭버튼 우측에 좌표 표출
        self.F2_L_V.setText(M_txt)
        self.check_save_point()

    def Save_position_2_R(self):
        self.F2_R_Position = self.F2_position
        M_txt = "X={0}, Y={1}".format(self.F2_R_Position[0],self.F2_R_Position[1])      #bub1 클릭버튼 우측에 좌표 표출
        self.F2_R_V.setText(M_txt)
        self.check_save_point()

    def check_save_point(self):                                     # 좌표 입력 되었는지 확인 각 좌표 입력 완료시 각 모델에 전달
        if self.F1_L_Position[0] * self.F1_L_Position[1] * self.F1_R_Position[0] * self.F1_R_Position[1] == 0:
            self.F1_ready = 0
        else:
            self.F1_ready = 1
            self.Watcher1.set_line_position(self.F1_L_Position[0],self.F1_L_Position[1],self.F1_R_Position[0],self.F1_R_Position[1])
            
            self.Watcher1.ready = 1

        if self.F2_L_Position[0] * self.F2_L_Position[1] * self.F2_R_Position[0] * self.F2_R_Position[1] == 0:
            self.F2_ready = 0
        else:
            self.F2_ready = 1
            self.Watcher2.set_line_position(self.F2_L_Position[0],self.F2_L_Position[1],self.F2_R_Position[0],self.F2_R_Position[1])
            self.Watcher2.ready = 1

class Fh_watcher(QThread):
    
    ImageUpdate = pyqtSignal(QImage)                #ImageUpdate 라는 시그널 발생 : 가지고가는 데이터 = QImage
    dist_num = pyqtSignal(float)                    #dist_num 시그널 : float가져감
    log_time = pyqtSignal(float)                    #로그 기록할 타임에 시그널줌    쓸지말지 메인에서 가져가는걸로 할까??
    
    def __init__(self, parent,fh_yolo_model_path,fh_rtsp_url):
        super().__init__(parent)
        self.fh_yolo = DW_detect_fh(fh_yolo_model_path)
        self.ready = 0                              #해당영사엥 좌표 모두 준비돠었나?
        self.Fh_num = 0                             #몇로 영상인가?
        self.RTSP_url = fh_rtsp_url                          #영상 url
        self.YOLO_model_path = ""                   #모델 pt파일 경료
        self.img_mul =1                             #기본 이미지 배율
        self.near_dist = 0                          #현재 최단거리
        self.no_img_dist = 150                      #배치 미검출시 사용할 거리 (버블러기준 투입구쪽 픽셀 추정거리)
        self.add_weight_alpha = 0.05
        self.line_info = [0,0,0]
        self.box_mid = ""

    
    def run(self):
        self.ThreadActive = True                                            #?
        
        Capture = cv2.VideoCapture(self.RTSP_url)
        self.video_height = int(Capture.get(cv2.CAP_PROP_FRAME_HEIGHT))     #동영상 높이
        self.video_width = int(Capture.get(cv2.CAP_PROP_FRAME_WIDTH))       #동영상 넓이
        t0 = time.time()

        while self.ThreadActive:
            try:
                self.process()
            except Exception as e: 
                print(e)
                print('중단 다시')

    def process(self):
        first_process = 0
        Capture = cv2.VideoCapture(self.RTSP_url)
        ret, dst = Capture.read()
        
        print('preocess start and ret->',ret)
        t0 = time.time()
        pred_opt = 0
        while ret:
            #print('----')
            ret, frame = Capture.read()
            if test_state:
                frame = origin_img


            t1 = time.time()
            dst = cv2.addWeighted(frame, self.add_weight_alpha, dst, 1-self.add_weight_alpha, 0)


            if first_process == 0 and t1-t0 > 10:       #처음실행
                first_process =1
                pred_opt = 1
                t0 = time.time()
                prep_img = fh2_img_const_zero_bler(dst,8)
                _, pred = self.fh_yolo(prep_img)

            if t1-t0 > 29:                             #이후실행
                pred_opt = 1
                t0 = time.time()
                prep_img = fh2_img_const_zero_bler(dst,8)
                _, pred = self.fh_yolo(prep_img)
                if self.ready:                             #거리계산
                    self.near_dist = self.dist_point_line2(pred,self.line_position_DIST[0], self.line_position_DIST[1] )
                    self.dist_num.emit(self.near_dist)
                    
                

            if pred_opt:                                        #객체 표시
                Image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                for i in pred:
                    pt1 = (int(i[0]),int(i[1]))
                    pt2 = (int(i[2]),int(i[3]))
                    Image = cv2.rectangle(Image,pt1,pt2,(255,255,0),2)
            else:   
                Image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)        

            if self.ready:      #q버블러 라인 표시
                Image = cv2.line(Image, self.line_position_DIST[0], self.line_position_DIST[1], (0,0,255), thickness=1, lineType=cv2.LINE_AA)                        #라인 색상, 두께 타입설정

            ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(int(round(self.img_mul*self.video_width,0)), int(round(self.img_mul*self.video_height,0)), Qt.KeepAspectRatio)   #이미지 배율만큼 변환
            self.ImageUpdate.emit(Pic)                                  # 이미지 바꾸라는 시그널

    def dist_point_line2(self, pred_result, set1, set2 ):                   #거리구하는 함수
        if pred_result.numel() == 0:
            end = self.no_img_dist
        
        else:
            a = (set2[1]-set1[1])/(set2[0]-set1[0])                                #직선의 방정식 ax + by+ c = 0
            b = -1
            c = -a*set1[0] + set1[1]   

            a,b,c = self.line_info

            target = (pred_result[:,[0,1]] + pred_result[:,[2,3]])/2                #pred 박스의 중심점
            target_x = target[:,0] 
            target_y = target[:,1]

            #self.box_mid = [target_x, target_y]

            dist = (a*target_x + b*target_y +c)/ math.sqrt(a*a + b*b)             #점과 직선의 거리  원점(좌측 하단)      
            
            end = torch.min(dist) 
            print(end)                                                  #최소값
        
        return float(end)      

    def set_line_position(self,LX,LY,RX,RY):
        self.line_position_DIST = [(LX, LY), (RX,RY)]
        a = (RY - LY) / ( RX - RY ) 
        b = -1
        c = -a*LX + LY
        # ax + by + c = 0
        self.line_info = [a,b,c]

class Fh_alarm(QThread):
    alarm_btn_signal = pyqtSignal(str)              #init에 셀프로 올리면 문제생김 이유 모름
    alarm_background_signal = pyqtSignal(str)       #init에 셀프로 올리면 문제생김 이유 모름

    def __init__(self, parent):
        super().__init__(parent)
        self.Alarm_state = 0                                                 #알람상태 0= 정상, 1 = 배치넘어옴, 2 =알람종료상태
        self.f1_dist = 0
        self.f2_dist = 0
        self.Alarm_stop_count = 0

    def run(self):
        
        self.ThreadActive = True
        while self.ThreadActive:
            if self.Alarm_state == 0:                                           #알람상태가 0이면 <시작은 0>
                self.alarm_btn_signal.emit("내부 정상 0")                            #버튼 표시 정상화
                if (self.f1_dist < 0 )or ( self.f2_dist < 0):                                   #정상인데 배치 넘어오는 확인 시 
                    self.Alarm_state = 1                                                #알람상태 1로 변경

            elif self.Alarm_state == 1:                                         #알람 상태가 1이면
                print("알람발생 코드 입력")
                self.alarm_btn_signal.emit("배치 상태 이상 1 (알람정지)")             #알람 버튼에 이상표시 표출
                self.alarm_background_signal.emit("1")                              #배경 무작위 색상 표출
                
                if (self.f1_dist > 0) and (self.f2_dist > 0):                                   #정상이 되면
                    self.Alarm_state = 0                                                #알람상태 1로 변경
                    self.alarm_background_signal.emit("0")                              #배경상태 정상으로 변경

            elif self.Alarm_state == 2:                                            #알람 상태가 2가되면 (알람 정지 버튼을 눌렀다는 소리)
                txt = "알람 정지2 / -" + str(15*60-self.Alarm_stop_count)
                self.alarm_btn_signal.emit(txt)                                    #알람버튼 정지표시 표출
                self.Alarm_stop_count += 1                                         #알람상태 지속시간 추가
                if self.Alarm_stop_count > 15*60 :                                 #알람상태 정지시간 지나면 (현재 15분으로 설정)
                    self.Alarm_stop_count = 0                                      #알람 지속시간 초기화
                    self.Alarm_state = 0                                           #알람상태 0 으로 변경


            if self.Alarm_state == 1:                                           #알람상태가 1이면
                winsound.Beep(2500, 1000)                                       #윈도우 신호음 1초간 전달
                                                             
            time.sleep(1)

class DW_detect_fh(DW_detect):
    def __init__(self, strWeightPath):                                              #사실 AI모델상 변화는 거의 없기에 있는모델로 썻어도 되지 않았을까 함
        super().__init__(strWeightPath)
        #self.device = select_device('cpu')               #cpu test
        self.conf_thres=0.4

    def __call__(self,img):
        yolo_img, pred = self.AI_detect(img)
        return yolo_img, pred

if __name__ == "__main__":
    try:
        App = QApplication(sys.argv)
        print('pase 1')
        Root = MainWindow()
        print('pase 2')

        Root.show()
        print('pase 3')

        sys.exit(App.exec())
    except Exception as e:
        print(e)