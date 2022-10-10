# https://rfriend.tistory.com/497 pandas timestamp 찍는법

# git fetch --all                       #깃 저장소 로컬에 덮어쓰기
# git reset --hard origin/master        #깃 저장소 로컬에 덮어쓰기


#   git config --global user.email "gusrms4650@naver.com"
#   git config --global user.name "LHG"


from PyQt5 import uic

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui

import pandas as pd
import sys
import matplotlib.pyplot as plt
import random
import datetime

# %matplotlib inline

plt.rcParams['font.family'] = 'Malgun Gothic'  # <<< 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False  # <<< -(마이너스) 부호 깨짐 방지


ui_path = "data/gui_v1.ui"
sen_path = "data/sentence.xlsx"

main_window_form_class = uic.loadUiType(ui_path)[0]


class MainWindow(QtWidgets.QMainWindow, main_window_form_class):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # self.check_inocent():
        # self.pushButton.clicked.connect(self.get_new_sen)
        self.get_new_sen()
        self.eng_key = 1

    # def check_inocent(self):
    #     db = pd.read_csv(sen_path,encoding = 'cp949').set_index('key')
    #     db[time]

    def get_new_sen(self):
        self.db = pd.read_excel(sen_path).set_index('key')
        self.db['time'] = pd.to_datetime(self.db['time'])


        db = self.db
        M_count = (db['correct']==1).sum()                              # 외운 갯수
        UnM_count = (db['correct']!=1).sum()                            # 못외운 개수
        dif = (datetime.datetime.now() - db['time']).mean()

        if pd.isnull(dif):
            dif = datetime.timedelta(0.001)
        time = dif.total_seconds()/60/60/24
        
        try:
            time = str(int(time))+"일 "+str(int((time-int(time))*24))+"시간"
        except:
            time = '0일 0시간'
    
        print(time)
        status = '외운 문장:',M_count,'/외우는 중:',UnM_count,'/암기주기:',time      
        self.status_view.setText(str(status)[1:-1])

        if M_count > 50:
            older_M_key = db['time'].sort_values()[:int(M_count*.2)].index.tolist()      # 외운것중 오래된것들 추출 10개
        else:
            older_M_key = db['time'].sort_values()[:10].index.tolist()      # 외운것중 오래된것들 추출 10개

        upper_UnM_key = db[db['correct']!=1][:10].index.tolist()

        try:
            get_setting = int(self.intval_set.text())
        except:
            get_setting = 1
        mem_standard_interval = datetime.timedelta(get_setting)

        if UnM_count > 0:
            if dif > mem_standard_interval*2:
                print('복습모드')
                mode = '복습'
                print('외운것중 오래된것중 1개')
                target = random.choice(older_M_key)

            elif dif > mem_standard_interval:
                print('암기 + 복습모드')
                mode = '암기 + 복습'
                print('외운것중 오래된거나 안외운것중에 최근거 5개중 1개')
                Mem = random.choice(older_M_key)
                UnM = random.choice(upper_UnM_key[:5])
                target = random.choice([Mem,UnM])

            elif dif < mem_standard_interval:
                print('암기모드')
                mode = '암기'
                print('안외운것충 위에있는거중 10개중 1개')
                target =  random.choice(upper_UnM_key)
        
        else:
            print('복습모드')
            mode = '복습'

            print('외운것중 오래된것중 1개')
            target = random.choice(older_M_key)

        self.mode_state.setText(mode)


        self.target = target
        self.target_data = db.loc[self.target]
        self.Ko_window.setText(str(self.target_data['ko']))
        self.Eng_window.setText("")

        self.sen_info.setText(str(self.target_data['part']))
        self.eng_key = 1

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key_A:    # A누르면 정답
            self.save_recode(1)
            self.get_new_sen()
            self.Eng_window.setText('')

        elif a0.key() == Qt.Key_D:  # D 틀림
            # self.save_recode(0)
            self.get_new_sen()
            self.Eng_window.setText('')

        elif a0.key() == Qt.Key_S:  # S 누르면 정답을 보여줘라
            if self.eng_key:
                self.Eng_window.setText(str(self.target_data['eng']))
                self.eng_key = 0
            else:
                self.Eng_window.setText("")
                self.eng_key = 1

    def save_recode(self, num):
        time_stamp = datetime.datetime.now()

        self.db['time'].loc[self.target] = time_stamp
        self.db['correct'].loc[self.target] = num
        self.db.to_excel(sen_path)  # , encoding='cp949')


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
