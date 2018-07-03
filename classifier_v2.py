import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets, QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import os
from os import listdir
from os.path import isfile, join
import fnmatch
from random import shuffle
import time
import _pickle as pickle
import numpy as np
import scipy.misc as misc
from sklearn import svm
from sklearn.svm import SVC
import sklearn
import pickle
import tensorflow as tf
from tqdm import tqdm
import argparse
import load_features as load
from compute_features import compute_img_features
from load_features import load_img_features
import time
from sklearn import linear_model

class classifier_v2(QMainWindow):
    def __init__(self):
        super().__init__()
        css = """QWidget{
            Background: #efd499;
            color:black;
            font:12px bold;
            font-weight:bold;
            border-radius: 1px;
            height: 11px;
            }
            QPushButton{
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #9e8550, stop: 1; #847457);
            border-radius: 15px;
            border: 1px;
            border-style: outset;
            border-width: 2px;
            border-color: black;
            font: 13px;
            padding: 2px;
          }
            QPushButton:pressed {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #847457, stop: 1 #9e8550);
            border-style: inset;
        }
        """
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QtGui.QPalette.Highlight)
        self.setStyleSheet(css)
        self.initUI()

    def initUI(self):
        self.statusBar().showMessage('No Images Loaded')
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
        self.qbtn = QPushButton("Quit",self)
        self.qbtn.clicked.connect(self.save)
        self.qbtn.resize(50,20)
        self.qbtn.move(200,400)
        self.next = QPushButton("Next",self)
        self.next.resize(50,20)
        self.next.move(275,400)
        self.next.clicked.connect(self.getNext)
        self.prev_btn = QPushButton("Previous",self)
        self.prev_btn.resize(70,20)
        self.prev_btn.move(350,400)
        self.prev_btn.clicked.connect(self.getPrev)
        self.load = QPushButton("Load Images",self)
        self.load.resize(100,20)
        self.load.move(440,400)
        self.load.clicked.connect(self.init_load)
        self.skip = QPushButton("Skip",self)
        self.skip.resize(50,20)
        self.skip.move(560,400)
        self.skip.clicked.connect(self.skip_img)
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
        self.images= 0                          #tracks number of images
        self.feats = None
        self.imag_reps = []                     #classified representations
        self.un_prev = []                     #previous unclassified
        self.class_vals = []                    #classified indexes
        self.classA_list = []
        self.classB_list = []
        self.rec = []                           #recently classified indexes
        self.skipped = []                       #indexes of skipped images
        self.img_dict = {}                      #dictionary of {index:images}
        self.npy_dict = {}                      #dic of numpy array reps of images
        self.paths = []                         #list of all image paths
        self.full_paths = []                    #holds the pkl file for feat reps (if found)
        self.index = 1                          #image index (starts at 1)
        self.path = None                        #path to directory of images
        self.d = {}                             #dictionary of {image:label}
        self.learn_type = "random"              #actove learning method (default = random)
        self.first_time = False                 #only for a certain case when partial-fit does not have enough data
        self.clf = linear_model.SGDClassifier()
        self.skip_flg = False
        self.path_len = 0
        self.prev = -1                          #index of previous image
        self.k = QShortcut(QKeySequence("1"),self)
        self.k.activated.connect(self.classB_event)
        self.shtct = QShortcut(QKeySequence("2"),self)
        self.shtct.activated.connect(self.classA_event)
        self.skct = QShortcut(QKeySequence("3"),self)
        self.skct.activated.connect(self.skip_event)
        self.skipped = []

        combo.move(25, 400)
        self.lbl.move(25, 370)
        combo.activated[str].connect(self.chooseFeats)

        lbl2 = QLabel("Active Learning Method:",self)
        combo2 = QComboBox(self)
        combo2.addItem("Farthest")
        combo2.addItem("Closest")
        combo2.addItem("Random")
        combo2.activated[str].connect(self.chooseModel)

        combo2.move(25, 325)
        lbl2.move(25, 300)
        self.show()




    def chooseFeats(self,value):
        self.statusBar().showMessage('Loading Features')
        if self.classA_list == [] and self.classB_list == [] and value != 'pixels':
            type = value
            path = self.paths
        if self.check_and_reload() == False:
            self.first_time = True
            self.statusBar().showMessage('Loading Features...')
            if value != "Pixels":
                for i in self.full_paths:
                    if value in i:
                        self.statusBar().showMessage('Reloading Features...')
                        self.feats = load_img_features(type,self.path)
                        self.makeData(self.feats)
                        self.statusBar().showMessage('Ready')
                        break
                if self.feats == None:
                    self.statusBar().showMessage('Loading Feature Data...')
                    self.statusBar().showMessage('Computing Features...')
                    compute_img_features(type,path,self.path)
                    self.statusBar().showMessage('Loading Features')
                    self.feats = load_img_features(type,self.path)
                    self.makeData(self.feats)
            else:
                self.statusBar().showMessage("Computing Pixel Features...")
                self.load_pix_features()
                self.statusBar().showMessage('Ready')
            self.path_len = len(self.paths)
            self.load_img()

    def check_and_reload(self):
        self.statusBar().showMessage('Loading Data...')
        if os.path.isfile(self.path+'/labels.pkl'):
           self.statusBar().showMessage('Reloading Previous Data...')
           self.d, self.classA_list, self.classB_list,self.skipped,self.img_dict,self.npy_dict,self.paths,self.imag_reps,self.class_vals,self.un_prev= pickle.load(open(self.path+'/labels.pkl', 'rb'))
           for i in self.d:
               if self.d[i] == 1 or self.d[i] ==2:
                   self.images+=1
           self.statusBar().showMessage('Checking Images...')
           if len(self.classA_list) < 1 or len(self.classB_list) < 1:
               self.first_time = True
           if self.classA_list != [] or self.classB_list !=[]:
               while(self.index in self.classA_list or self.index in self.classB_list or self.index in self.skipped):
                   self.index +=1
                   if self.index > len(self.paths):
                       self.statusBar().showMessage('All Images Classified!')
                       self.save()
               self.statusBar().showMessage('Fitting Data...')
               self.clf.fit(np.asarray(self.imag_reps),np.asarray(self.class_vals))
               self.statusBar().showMessage('Loading Image...')
               self.path_len = len(self.paths)
               self.load_img()
               self.statusBar().showMessage('Ready')
               return True
        return False


#should only occur once at the beginning to match up all of the Picture data
    def makeData(self,feats):
        self.statusBar().showMessage('Syncing Data...')
        if feats != None:
            index = 1
            self.paths = []
            self.npy_dict = {}
            self.img_dict = {}
            for i in feats:
                self.paths.append(i)
                self.img_dict[index] = i
                self.npy_dict[index] = feats[i]
                index +=1

    def chooseModel(self,value):
        if value == "Random":
            self.learn_type = "r"
        elif value == "Closest":
            self.learn_type = "c"
        elif value == "Farthest":
            self.learn_type = "f"

    def getNext(self):
        c = self.get_unclassified()
        if c != ([],[]):
            if self.learn_type == "r" or (self.classA_list == [] or self.classB_list == []):
                index = self.index
                index +=1
                if index < len(self.paths):
                    self.index = index
                    self.load_img()
                else:
                    print("All Images Classified")
                    self.save()

            # if both lists have contents, train svm here
            elif self.learn_type == "c":
                if self.first_time:
                    self.clf.fit(np.asarray(self.imag_reps),np.asarray(self.class_vals))
                    self.index +=1
                    self.prev = self.index
                    self.load_img()
                    self.first_time = False

                else:
                    self.prev = self.index
                    unclass_vals,indexes = c
                    unclass_vals = np.asarray(unclass_vals)
                    temp = np.array([self.imag_reps[-1]])
                    self.clf.partial_fit(temp,np.asarray(self.class_vals)[-1])
                    if len(unclass_vals) == 1:
                        unclass_vals = unclass_vals.reshape(1,-1)
                        self.index = indexes[0]
                        self.load_img
                    if len(unclass_vals) > 1:
                       closest = np.argmin(self.clf.decision_function(unclass_vals))
                       self.index = indexes[closest]
                       self.load_img()
            else:
                if self.first_time:
                    self.clf.fit(np.asarray(self.imag_reps),np.asarray(self.class_vals))
                    self.prev = self.index
                    self.index +=1
                    self.load_img()
                    self.first_time = False
                else:
                    self.prev = self.index
                    unclass_vals,indexes = c
                    unclass_vals = np.asarray(unclass_vals)
                    temp = np.array([self.imag_reps[-1]])
                    s = time.time()
                    self.clf.partial_fit(temp,np.array([self.class_vals[-1]]))
                    e = time.time()
                    if len(unclass_vals) == 1:
                        unclass_vals = unclass_vals.reshape(1,-1)
                        self.index = indexes[0]
                        self.load_img
                    if len(unclass_vals) > 1:
                        farthest = np.argmax(self.clf.decision_function(unclass_vals))
                        self.index = indexes[farthest]
                        self.load_img()
        else:
            self.save()


    def classA_event(self):  #class A Event (User Presses 1)
        self.images+=1
        self.d[self.paths[self.index-1]] = 2
        self.skip_flg = False
        self.rec.append(self.index)
        self.images+=1
        self.numImages.setText(str(self.images))
        self.classA_list.append(self.index)
        self.imag_reps.append(self.npy_dict[self.index])
        self.class_vals.append(2)
        self.getNext()
    def classB_event(self): #class B Event (User Presses 2)
        self.images+=1
        self.d[self.paths[self.index-1]] = 1
        self.skip_flg = False
        self.rec.append(self.index)
        self.images+=1
        self.numImages.setText(str(self.images))
        self.classB_list.append(self.index)
        self.imag_reps.append(self.npy_dict[self.index])
        self.class_vals.append(1)
        self.getNext()
    def skip_event(self): #class Skip Event (User Presses 3)
        self.d[self.paths[self.index-1]] = -1
        self.skip_flg = True
        self.rec.append(self.index)
        self.skipped.append(self.index)
        self.prev = self.index
        self.index = 1

        while self.index in self.classA_list or self.index in self.classB_list or self.index in self.skipped:
            self.index +=1
        if self.index > self.path_len:
            print("All Images Classified")
        else:
            self.load_img()

#first case should only occur once at the beginning
    def get_unclassified(self):
        if self.un_prev == None:
            unclass = []
            indexes = []
            for i in self.npy_dict:
                if i not in self.classA_list and i not in self.classB_list and i not in self.skipped:
                    unclass.append(self.npy_dict[i])
                    indexes.append(i)
            #print(len(unclass))
            self.un_prev = unclass,indexes
            self.rec = []
            return unclass,indexes
        else:
            index = 0
            unclass,indexes = self.un_prev
            for i in indexes:
                if i in self.rec:
                    indexes.remove(i)
                    del unclass[index]
                index+=1
            #print(len(unclass))
            self.rec = []
            self.un_prev = unclass,indexes
            return unclass,indexes

    def load_pix_features(self):
         self.feats =[]
         for e,p in enumerate(self.paths):
             self.feats.append(misc.imresize(misc.imread(p), (self.height, self.width, 3)))
         self.feats = np.asarray(self.features)
         self.makeData(self.feats)
         self.load_img()

    def getPrev(self):
        self.images-=1
        if self.learn_type == 'r' or (self.classA_list == [] or self.classB_list == []):
            self.d[self.paths[self.index-2]] = 0
            index = self.index
            index -=1
            if index > 0:
                self.index = index
                self.load_img()
            if self.index in self.classA_list:
                self.classA_list.remove(self.index)
            elif self.index in self.classB_list:
                self.classB_list.remove(self.index)
            else:
                self.skipped.remove(self.index)
        elif self.learn_type in "cf":
            if self.skip_flg:
                self.skipped.remove(self.prev)
                self.rec.remove(self.prev)
                self.index = self.prev
                self.d[self.paths[self.index-1]] = 0
                self.load_img()
            else:
                self.index = self.prev
                self.imag_reps.reverse()
                self.class_vals.reverse()
                self.imag_reps = self.imag_reps[1:]
                self.imag_reps.reverse()
                self.class_vals = self.class_vals[1:]
                self.class_vals.reverse()
                self.d[self.paths[self.index-1]] = 0
                if self.index in self.classA_list:
                    self.classA_list.remove(self.index)
                else:
                    self.classB_list.remove(self.index)
                self.load_img()


    def init_load(self):
        self.statusBar().showMessage('Loading Images...')
        self.path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.paths = self.getPaths(self.path)
        self.statusBar().showMessage('Images Loaded')
        self.statusBar().showMessage('Choose Image Features')
        self.label.setText("Choose Image Features")

    def load_img(self):
        if self.index < len(self.paths):
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
        exts = ['*.pkl']
        for pattern in exts:
            for d, s, fList in os.walk(data_dir):
                for filename in fList:
                    if fnmatch.fnmatch(filename, pattern):
                        fname_ = os.path.join(d,filename)
                        self.full_paths.append(fname_)
        return self.paths

    def skip_img(self):
        print("not yet implemented")

    def save(self):
        if self.get_unclassified() == ([],[]) and self.paths != []:
            self.statusBar().showMessage('All Images Classified')
        else:
            self.statusBar().showMessage('Closing...')
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.statusBar().showMessage('Saving Data...')
            if self.rec != []:
                self.un_prev = self.get_unclassified()
            if self.classA_list != [] or self.classB_list != []:
                pkl = open(self.path+'/labels.pkl', 'wb')
                data = pickle.dumps([self.d, self.classA_list, self.classB_list,self.skipped,self.img_dict,self.npy_dict,self.paths,self.imag_reps, self.class_vals,self.un_prev])
                pkl.write(data)
                pkl.close()
            self.statusBar().showMessage('Exiting...')
            exit()

app = QApplication(sys.argv)
c = classifier_v2()
sys.exit(app.exec_())
