import sys
import os
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from yolo import YOLO
import os
from langdetc import LangNet,DetectLan
from NER import NER
from PIL import Image

from Client import *

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.pixmap = QPixmap()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.Open_Dir_button.clicked.connect(lambda: self.getDirectory())
        self.ui.Open_file_button.clicked.connect(lambda: self.getFileName())
        self.ui.Close_button.clicked.connect(lambda: self.showMinimized())
        self.ui.Exit_button.clicked.connect(lambda: self.close())
        self.ui.Max_button.clicked.connect(lambda: {self.restore_or_maximize_window(),self.ratio()})
        self.ui.Next_page.clicked.connect(lambda :self.next_image())
        self.ui.Prev_page.clicked.connect(lambda :self.previous_image())
        self.ui.Delete_button.clicked.connect(lambda: self.delite_page())
        self.ui.Start_button.clicked.connect(lambda: {self.print_smth("Starting..."),self.start()})
        self.ui.Save_button.clicked.connect(lambda :self.save_data())
        self.Image_list = []
        self.filename =''
        self.number_of_pages =0
        self.temp_page=0
        self.ui.Side_button.clicked.connect(lambda: self.slideLeftMenu())
        self.show()
        self.dirlist =""




    def slideLeftMenu(self):
        width = self.ui.Side_bar_container.width()
        if width == 0:
            newWidth = 100
            self.ui.Side_button.setIcon(QtGui.QIcon(u"Data/icons/icons/chevron-left.svg"))
        else:
            newWidth = 0
            self.ui.Side_button.setIcon(QtGui.QIcon(u"Data/icons/icons/align-left.svg"))

        self.animation = QPropertyAnimation(self.ui.Side_bar_container, b"maximumWidth")
        self.animation.setDuration(250)
        self.animation.setStartValue(width)
        self.animation.setEndValue(newWidth)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()
        print(self.ui.Doc_img.size())


    def getDirectory(self):
        self.dirlist = QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")
        if self.dirlist!="":
            self.number_of_pages=0
            self.Image_list.clear()
            self.ui.listWidget.addItem("Open: "+self.dirlist)
            for i in os.listdir(self.dirlist):
                temp = i[-4:]
                if temp==".jpg" or temp==".png":
                    self.number_of_pages+=1
                    self.Image_list.append(self.dirlist+'/'+i)
                    self.ui.listWidget.addItem("Upload: "+self.Image_list[self.number_of_pages-1])
            if self.number_of_pages!=0:
                self.temp_page=1
                self.file_init()

    def next_image(self):
        if self.number_of_pages>1:
            if self.temp_page<self.number_of_pages:
                self.temp_page+=1
            else:
                self.temp_page = 1
            self.file_init()

    def previous_image(self):
        if self.number_of_pages>1:
            if self.temp_page>1:
                self.temp_page-=1
            else:
                self.temp_page = self.number_of_pages
            self.file_init()
        elif self.number_of_pages==1:
            self.temp_page = self.number_of_pages
            self.file_init()

    def delite_page(self):

        if self.Image_list:
            self.Image_list.pop(self.temp_page-1)
            self.number_of_pages-=1
            if self.number_of_pages==0:
                self.ui.Doc_name.clear()
                self.ui.Doc_img.clear()
                self.ui.Page_number.setText("{} / {}""".format(0, self.number_of_pages))
            if self.temp_page+1<=self.number_of_pages:
                self.temp_page+=1
            self.previous_image()


    def getFileName(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self,"Выбрать файл", ".","Img Files(*.jpg *.png *.pdf)")
        print(self.filename)
        if self.filename!="":
            self.number_of_pages=1
            self.temp_page=1
            self.Image_list.clear()
            self.Image_list.append(self.filename)
            self.ui.listWidget.addItem("Upload: "+self.filename)
            self.file_init()

    def start(self):
        yolo = YOLO()
        file = open("temp2.txt", 'w', encoding="UTF-8")
        for i in self.Image_list:
            try:
                img = Image.open(i)
            except:
                self.ui.listWidget.addItem(i+' Open Error!')
                continue
            else:
                self.ui.listWidget.update()
                str = i+"\n"+"{\n"
                file.write(str)
                r_image = yolo.detect_image(i, DetectLan(i))
                NER(file)
                file.write("}\n\n")
                self.ui.listWidget.addItem(i+' Successfully Detected')


    def save_data(self):
        response = QFileDialog.getSaveFileName(self,"","Save.txt")
        f = open("temp2.txt",'r',encoding='UTF-8')
        ff = open(response[0],'w',encoding='UTF-8')
        for line in f:
            ff.write(line)
        self.ui.listWidget.addItem("Saved!")
    def print_smth(self,str):
        self.ui.listWidget.addItem(str)


    def mousePressEvent(self, event):
        self.clickPosition = event.globalPos()


    def restore_or_maximize_window(self):
        if self.isFullScreen():
            self.showNormal()
            self.ui.Max_button.setIcon(QtGui.QIcon(u"Data/icons/icons/maximize-2.svg"))
        else:
            self.showFullScreen()
            self.ui.Max_button.setIcon(QtGui.QIcon(u"Data/icons/icons/minimize-2.svg"))


    def ratio(self):
        scaled = self.pixmap.scaled(self.ui.Doc_img.size(), Qt.KeepAspectRatio)
        self.ui.Doc_img.setPixmap(scaled)

    def file_init(self):
        self.filename = self.Image_list[self.temp_page - 1]
        self.ui.Doc_name.setText(self.filename)
        self.ui.Page_number.setText("{} / {}""".format(self.temp_page, self.number_of_pages))
        self.pixmap = QPixmap(self.filename)
        self.ratio()







if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print(0)
    sys.exit(app.exec_())