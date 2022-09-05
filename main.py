# https://rfriend.tistory.com/497 pandas timestamp 찍는법


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
        # , encoding = 'cp949'), encoding='UTF-8')
        # , encoding='cp949').set_index('key')
        self.db = pd.read_excel(sen_path).set_index('key')
        self.db['time'] = pd.to_datetime(self.db['time'])

        if self.db[self.db['correct'] == 0].__len__() != 0:
            target_db = self.db[self.db['correct'] == 0]
            print(target_db.index.tolist().__len__(), '0 exist')
            self.target = random.choice(target_db.index.tolist())
        else:
            print('1 exist')
            target_db = self.db[self.db['correct'] == 1]
            target_db = target_db.sort_values('time')
            target_idx_list = target_db.index.tolist()[
                :int(self.db.__len__()*.2)]
            self.target = random.choice(target_idx_list)

        print(self.target)
        self.target_data = target_db.loc[self.target]
        self.Ko_window.setText(str(self.target_data['ko']))
        self.Eng_window.setText("")
        self.eng_key = 1

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key_A:    # A누르면 정답
            print('정상')
            self.save_recode(1)
            self.get_new_sen()
            self.Eng_window.setText('')

        elif a0.key() == Qt.Key_D:  # D 틀림
            print('정상')
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
