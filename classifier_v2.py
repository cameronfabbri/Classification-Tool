import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton,QMessageBox,QAction, qApp,QFileDialog, QLabel,QLCDNumber, QSlider,QVBoxLayout,QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtWidgets, QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import os
from os import listdir
from os.path import isfile, join
import fnmatch
from random import shuffle

class classifier_v2(QMainWindow):
    def __init__(self):
        super().__init__()
        css = """QWidget{
            Background: #C9CACA;
            color:black;
            font:12px bold;
            font-weight:bold;
            border-radius: 1px;
            height: 11px;
        }
        QPushButton{
        background-color: #BDB6B7;
        border-style: round;
        border: 3px;
        font: 13px;
        padding: 2px;
      }
        """
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QtGui.QPalette.Highlight)
        self.setStyleSheet(css)
        self.initUI()
        self.path = ""
        self.prev = -1
        self.paths = []
        self.index = 1
        self.images= 0

    def initUI(self):


        self.statusBar().showMessage('Ready')
        self.label = QLabel("No Images Loaded",self)
        self.label.resize(300,300)
        self.label.move(340,50)
        self.dist = QLabel("1 : Distorted",self)
        self.dist.resize(100,30)
        self.dist.move(340,420)
        self.un = QLabel("2 : Non-Distorted",self)
        self.un.resize(120,30)
        self.un.move(340,440)
        self.sk = QLabel("3 : Skip Image",self)
        self.sk.resize(120,30)
        self.sk.move(340,460)
        self.numImages = QLabel("0",self)
        self.numImages.resize(self.label.sizeHint())
        self.numImages.move(150,400)
        self.setGeometry(0,0,700,500)
        self.setWindowTitle("Binary Picture Classifier")
        self.setWindowIcon(QIcon("download-1.png"))
        qbtn = QPushButton("Quit",self)
        qbtn.clicked.connect(QApplication.instance().quit)
        qbtn.resize(50,20)
        qbtn.move(200,400)
        next = QPushButton("Next",self)
        next.resize(50,20)
        next.move(275,400)
        next.clicked.connect(self.next)
        prev = QPushButton("Previous",self)
        prev.resize(70,20)
        prev.move(350,400)
        prev.clicked.connect(self.prev)
        load = QPushButton("Load Images",self)
        load.resize(100,20)
        load.move(440,400)
        load.clicked.connect(self.load)
        skip = QPushButton("Skip",self)
        skip.resize(50,20)
        skip.move(560,400)
        skip.clicked.connect(self.skip)
        self.lbl = QLabel("Image Features:",self)
        combo = QComboBox(self)
        combo.addItem("inception_v1")
        combo.addItem("inception_v2")
        combo.addItem("inception_v3")
        combo.addItem("inception_resnet_v2")
        combo.addItem("resnet_v1_50")
        combo.addItem("resnet_v1_101")
        combo.addItem("vgg_16")
        combo.addItem("vgg_19")

        combo.move(25, 400)
        self.lbl.move(25, 370)
        combo.activated[str].connect(self.onActivated)

        lbl2 = QLabel("Active Learning Method:",self)
        combo2 = QComboBox(self)
        combo2.addItem("Farthest")
        combo2.addItem("Closest")
        combo2.addItem("Random")
        combo2.activated[str].connect(self.onActivated_v2)

        combo2.move(25, 325)
        lbl2.move(25, 300)
        self.show()




    def onActivated(self):
        print("active")

    def onActivated_v2(self):
        print("activated")


    def selectionchange(self,i):
        print("Items in the list are :")
        for count in range(self.cb.count()):
            print(self.cb.itemText(count))
        print("Current index",i,"selection changed ",self.cb.currentText())

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def next(self):
        self.index +=1
        self.load_img()

    def keyPressEvent(self, e):
       print("b")
       #print(e.key())
       if e.key() == 49:   #class A Event (User Presses 1)
           print("B")
       elif e.key() == 50: #class B Event (User Presses 2)
           print("A")
       elif e.key() == 51: #class Skip Event (User Presses 3)
           print("S")


    def prev(self):
        self.index -=1
        self.load_img()

    def load(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.paths = self.getPaths(file)
        print(self.paths)
        self.load_img()

    def load_img(self):
        if self.index < len(self.paths):
            self.images+=1
            self.numImages.setText(str(self.images))
            pixmap = QPixmap(self.paths[self.index-1])
            self.label.move(320,50)
            self.label.setPixmap(pixmap)

    def getPaths(self,data_dir):
        exts = ['*.JPEG','*.JPG','*.jpg','*.jpeg','*.png','*.PNG']
        for pattern in exts:
            for d, s, fList in os.walk(data_dir):
                for filename in fList:
                    if fnmatch.fnmatch(filename, pattern):
                        fname_ = os.path.join(d,filename)
                        self.paths.append(fname_)
        shuffle(self.paths)
        return self.paths

    def skip(self):
        print("skipper")

app = QApplication(sys.argv)
c = classifier_v2()
sys.exit(app.exec_())
