from tkinter import *
from tkinter import filedialog
import os
from PIL import Image,ImageTk
import time
import _pickle as pickle
import fnmatch
import numpy as np
import scipy.misc as misc
from sklearn import svm
from sklearn.svm import SVC
import sklearn
from random import shuffle
import scipy.misc as misc
import pickle
import tensorflow as tf
from tqdm import tqdm
import argparse
import sys
from os import listdir
from os.path import isfile, join
import load_features as load
from compute_features import compute_img_features
from load_features import load_img_features
import time


# Active learning bit
'''
   http://scikit-learn.org/stable/modules/svm.html
   https://github.com/cameronfabbri/Compute-Features
   print np.linalg.norm(a)

   Load up images at start into a dictionary
   Step 1: Get random image
   Step 2: Keep getting random images until both classes are covered
   Step 3: When you have instances from both classes, then train an SVM with those classified
   Step 4: Calculate hyperplane (this is a function in sklearn)
   Step 5: if farthest: get image farthest from hyperplane
           if closest: get image closest to hyperplane
   Step 6: user classifies that, then SVM is updated every x classifications

'''

# this is how you load a pickle file. 'a' is then the dictionary that we saved
# pkl_file = open(sys.argv[1], 'rb')
# a = pickle.load(pkl_file)

class classifier():

    #sets up all buttons and binds keys for classification
    def __init__(self, root=None):
        self.width = 256
        self.height = 256
        self.img_list = []
        self.root = Tk()
#        self.root.style = Style()
#        self.root.style.theme_use("alt")
        #('clam', 'alt', 'default', 'classic')
        self.window = Frame(self.root)
        self.root.title("Picture Classifier")
        self.root.geometry('700x500')
        self.label1 = Label(self.root,text='1 = Distorted')
        self.label1.grid(column=0, row=3)
        self.label1 = Label(self.root,text='2 = Nondistorted')
        self.label1.grid(column=0, row=4)
        self.quit = Button(self.root,text = "Quit", command = self.save)
        self.quit.grid(column= 7,row = 5)
        self.next = Button(self.root,text = "Next", command = self.getNext)
        self.next.grid(column= 6, row = 5)
        self.prev = Button(self.root,text = "Previous", command = self.getPrev)
        self.prev.grid(column = 5, row = 5)
        self.load = Button(self.root,text = "Load", command = self.loadImages)
        self.load.grid(column = 8, row = 5)
        self.label = Label(text = "No Image Loaded",height = 15, width = 15)
        self.label.grid(row = 0, column = 1, columnspan = 10)
        self.numImages = Label(text = "0")
        self.numImages.grid(column = 0, row = 10)
        self.noclass = Button(self.root,text = "No Class", command = self.getNext)
        self.noclass.grid(column = 9, row = 5)
        choices = {'Random','Closest', 'Farthest'}
        modelChoices = ['pixels', 'inception_v1', 'inception_v2', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'vgg_16', 'vgg_19']

        self.dropVar = StringVar()
        self.modelVar = StringVar()
        self.dropVar.set("random")   #default
        self.modelVar.set('pixels')
        self.option_menu1 = OptionMenu(self.root,self.dropVar, *choices, command = self.func)
        self.option_menu2 = OptionMenu(self.root,self.modelVar, *modelChoices, command = self.choseModel)

        self.popup_label = Label(self.root, text= "Choose an Active Learning Method:").grid(row = 6, column = 0)
        self.popup_label = Label(self.root, text= "Choose image representation:").grid(row = 8, column = 0)
        self.option_menu1.grid(row = 7, column = 0)
        self.option_menu2.grid(row = 9, column = 0)

        try: self.initial_path = sys.argv[1]
        except: self.initial_path='./'

        # should contain the index of self.features for what was labeled
        self.imag_reps = []
        self.class_vals = []
        self.classA_list = [] # [2,5,1]
        self.classB_list = []
        self.skipped = []
        self.index = 1
        self.img_dict = {}
        self.paths = []
        self.npy_dict = {}
        self.model = "r"
        self.type = "None"
        self.clf = SVC(kernel='linear')
        self.feats = None
        self.full_paths= []
        self.images = 1

        def classA(event):
            self.images+=1
            self.numImages.config(text = str(self.images))
            self.classA_list.append(self.index)
            self.imag_reps.append(self.npy_dict[self.index])
            self.class_vals.append(2)
            self.getNext()
        def classB(event):
            self.images+=1
            self.numImages.config(text = str(self.images))
            self.classB_list.append(self.index)
            self.imag_reps.append(self.npy_dict[self.index])
            self.class_vals.append(1)
            self.getNext()
        def skipClassEvent(event):
            self.skipped.append(self.index)
            self.img_dict = self.delete_item(self.img_dict,self.index)
            self.npy_dict = self.delete_item(self.npy_dict,self.index)
            self.index = 1

            while self.index in self.classA_list or self.index in self.classB_list or self.index in self.skipped:
                self.index +=1
            self.load_img()

        self.root.bind(2, classA)
        self.root.bind(1, classB)
        self.root.bind(3, skipClassEvent)
        self.path = '/mnt/data1/'
        mainloop()

    '''
        loads all images (or features) on start up
        When using features, instead of reading images this will use the feature vector
    '''
    def loadImages(self):
        self.path = filedialog.askdirectory(initialdir=self.initial_path)
        self.prev = -1
        self.paths = self.getPaths(self.path)


    def delete_item(self,d,item):
        new = {}
        for i in d:
            #print(i)
            if  i != item:
                new[i] = d[i]
        return new



#    def load_pix_features(self):
#        self.features =[]
#        print('Loading images...')
#        for e,p in enumerate(self.paths):
#            self.features.append(misc.imresize(misc.imread(p), (self.height, self.width, 3)))
#        self.features = np.asarray(self.features)
#        print('Done')
#        self.check_and_reload()
#        self.make_npy_dict()
#        self.load_img()

    def choseModel(self,value):
        if self.classA_list == [] and self.classB_list == [] and value != 'pixels':
            type = value
            path = self.paths
        if self.check_and_reload() == False:
            for i in self.full_paths:
                if value in i:
                    self.feats = load_img_features(type,self.path)
                    self.remake_npy_dict(self.feats)
                    print("Images Reloaded")
                    break
            if self.feats == None:
                print("Loading Images")
                compute_img_features(type,path,self.path)
                print("Done")
                self.feats = load_img_features(type,self.path)
                self.remake_npy_dict(self.feats)
            self.load_img()


    def remake_npy_dict(self,new):
        index = 1
        self.paths = []
        self.npy_dict = {}
        self.img_dict = {}
        for i in new:
            self.paths.append(i)
            self.img_dict[index] = i
            self.npy_dict[index] = new[i]
            index +=1
        #self.numImages = len(self.paths)

    def check_and_reload(self):
        if os.path.isfile(self.path+'/labels.pkl'):
           d, self.classA_list, self.classB_list,self.skipped,self.img_dict,self.npy_dict,self.paths,self.imag_reps,self.class_vals = pickle.load(open(self.path+'/labels.pkl', 'rb'))
           print(len(self.skipped))
           #self.get_reps()
           self.images += len(self.classA_list) + len(self.classB_list)
           if self.classA_list != [] or self.classB_list !=[]:
               while(self.index in self.classA_list or self.index in self.classB_list or self.index in self.skipped):
                   self.index +=1
                   if self.index > len(self.paths):
                       print("All Images Classified")
                       self.save()
               self.load_img()
               return True
        return False

    def get_reps(self):
        for i in self.classA_list:
            self.imag_reps.append(self.npy_dict[i])
            self.class_vals.append(2)
        for i in self.classB_list:
            self.imag_reps.append(self.npy_dict[i])
            self.class_vals.append(1)
        print(self.class_vals)

    # preprocessing
    def func(self,value):
        if value == "Closest":
            self.model = "c"
        elif value == "Farthest":
            self.model = "f"
        else:
            self.model = "r"


    def get_unclassified(self):
        unclass = []
        indexes = []
        for i in self.npy_dict:
            if i not in self.classA_list and i not in self.classB_list and i not in self.skipped:
                unclass.append(self.npy_dict[i])
                indexes.append(i)
        return unclass,indexes



    #loads image on to the screen using a label
    def load_img(self):
        if self.index -1 < len(self.paths):
            print("index:",self.index)
            im = Image.open(self.paths[self.index-1])
            photo = ImageTk.PhotoImage(im)
            self.label.config(image=photo, height = self.height, width = self.width)
            self.label.image = photo




    def getPaths(self, data_dir):
        exts = ['*.JPEG','*.JPG','*.jpg','*.jpeg','*.png','*.PNG']
        for pattern in exts:
            for d, s, fList in os.walk(data_dir):
                for filename in fList:
                    if fnmatch.fnmatch(filename, pattern):
                        fname_ = os.path.join(d,filename)
                        self.paths.append(fname_)

        exts = ['*.JPEG','*.JPG','*.jpg','*.jpeg','*.png','*.PNG','*.pkl']
        for pattern in exts:
            for d, s, fList in os.walk(data_dir):
                for filename in fList:
                    if fnmatch.fnmatch(filename, pattern):
                        fname_ = os.path.join(d,filename)
                        self.full_paths.append(fname_)
        shuffle(self.paths)
        return self.paths



    #does not allow user to go back past the beginning the list of images
    def getPrev(self):
        if self.model == 'r' or (self.classA_list == [] or self.classB_list == []):
            index = self.index
            index -=1
            if index > 0:
                self.index = index
                self.load_img()
            if self.index in self.classA_list:
                self.classA_list.remove(self.index)
            else:
                self.classB_list.remove(self.index)
        elif self.model in "cf":
            self.index = self.prev
            self.imag_reps.reverse()
            self.class_vals.reverse()
            self.imag_reps = self.imag_reps[1:]
            self.imag_reps.reverse()
            self.class_vals = self.class_vals[1:]
            self.class_vals.reverse()
            self.load_img()
            if self.index in self.classA_list:
                self.classA_list.remove(self.index)
            else:
                self.classB_list.remove(self.index)




    #does not allow user to go past the end of the list
    def getNext(self):
        # if there is a trained svm, get next from here, otherwise random
        if self.get_unclassified() != ([],[]):
            if self.model == "r" or (self.classA_list == [] or self.classB_list == []):
                index = self.index
                index +=1
                if index < len(self.paths):
                    self.index = index
                    self.load_img()
                else:
                    print("All Images Classified")
                    self.save()

            # if both lists have contents, train svm here
            elif self.model == "c":
                if self.images %7 == 0:
                    self.prev = self.index
                    unclass_vals,indexes = self.get_unclassified()
                    unclass_vals = np.asarray(unclass_vals)
                    self.clf.fit(np.asarray(self.imag_reps),np.asarray(self.class_vals))
                    if len(unclass_vals) == 1:
                        unclass_vals = unclass_vals.reshape(1,-1)
                        self.index = indexes[0]
                        self.load_img
                    if len(unclass_vals) > 1:
                       closest = np.argmin(self.clf.decision_function(unclass_vals))
                       self.index = indexes[closest]
                       self.load_img()
                else:
                    self.prev = self.index
                    index = 1 
                    if self.index + 1 not in self.classA_list and self.index + 1 not in self.classB_list and self.index + 1 not in self.skipped and self.index + 1 < len(self.paths):
                        index = self.index + 1
                    else:
                        while index in self.classA_list or index in self.classB_list or index in self.classB_list or index in self.skipped:
                            index +=1
                    self.index = index 
                    self.load_img()
            else:
                if self.images %7 == 0: 
                    self.prev = self.index
                    unclass_vals,indexes = self.get_unclassified()
                    unclass_vals = np.asarray(unclass_vals)
                    self.clf.fit(np.asarray(self.imag_reps),np.asarray(self.class_vals))
                    if len(unclass_vals) == 1:
                        unclass_vals = unclass_vals.reshape(1,-1)
                        self.index = indexes[0]
                        self.load_img
                    if len(unclass_vals) > 1:
                        farthest = np.argmax(self.clf.decision_function(unclass_vals))
                        self.index = indexes[farthest]
                        self.load_img()
                else:
                    self.prev = self.index
                    index = 1 
                    if self.prev + 1 not in self.classA_list and self.prev + 1 not in self.classB_list and self.prev + 1 not in self.skipped and self.prev + 1 < len(self.paths):
                        index = self.prev + 1
                    else:
                        while index in self.classA_list or index in self.classB_list or index in self.classB_list or index in self.skipped:
                            index +=1
                    self.index = index 
                    self.load_img()
        else:
            print("All Images have been Classified")
            self.save()


    #happens each time the user presses quit
    #pushes all data out to text file
    #creates a unique name for each file
    def save(self):
        if self.classA_list != [] or self.classB_list != []:
            d = {}
            index = 1
            for i in self.paths:
                if index in self.classA_list:
                    d[i] = 2
                elif index in self.classB_list:
                    d[i] = 1
                elif index in self.skipped:
                    d[i] = -1
                else:
                    d[i] = 0
                index +=1

            pkl = open(self.path+'/labels.pkl', 'wb')
            data = pickle.dumps([d, self.classA_list, self.classB_list,self.skipped,self.img_dict,self.npy_dict,self.paths,self.imag_reps, self.class_vals])
            pkl.write(data)
            pkl.close()
        self.root.destroy()




c = classifier()
