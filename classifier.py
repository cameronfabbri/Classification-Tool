from tkinter import *
from tkinter import filedialog
import os
from PIL import Image,ImageTk
import time
import _pickle as pickle
import fnmatch
#from sklearn import svm

'''
   TODO in order of importance:
   x - Have a third button that is a "no class" if we don't want to use that image.
   - In the pickle file store class1 and class2 such that the user can set a label
   so we don't get confused whether 1 is distorted or 2 is distorted. When starting
   the program for the first time, check if this is set or not. If not, prompt the user
   to set it, otherwise just load the value and display it on the GUI somewhere.
   x - load previous pickle file
   x - Say which class is which for buttons 1 and 2
   x - drop down box selecting active learning method (current would be 'random')
   x - max image size so the image doesn't change window size
'''

# this is how you load a pickle file. 'a' is then the dictionary that we saved
# pkl_file = open(sys.argv[1], 'rb')
# a = pickle.load(pkl_file)

class classifier():

    #sers up all buttons and binds keys for classification
    def __init__(self, root=None):
        self.img_list = []
        self.root = Tk()
        self.window = Frame(self.root)
        self.root.title("Picture Classifier")
        self.root.geometry('500x300')
        self.label1 = Label(self.root,text='1 = Distorted')
        self.label1.grid(column=0, row=0)
        self.label1 = Label(self.root,text='2 = Nondistorted')
        self.label1.grid(column=0, row=1)
        self.quit = Button(self.root,text = "Quit", command = self.save)
        self.quit.grid(column= 3,row = 5)
        self.next = Button(self.root,text = "Next", command = self.getNext)
        self.next.grid(column= 2, row = 5)
        self.prev = Button(self.root,text = "Previous", command = self.getPrev)
        self.prev.grid(column = 1, row = 5)
        self.load = Button(self.root,text = "Load", command = self.get_path)
        self.load.grid(column = 4, row = 5)
        self.label = Label(text= "No Image Loaded",height = 15, width = 15)
        self.label.grid(row = 0, column = 1, columnspan = 10)
        self.noclass = Button(self.root,text = "No Class", command = self.getNext)
        self.noclass.grid(column = 5, row = 5)
        choices = {'Random','Closest', 'Farthest'}
        self.option_menu = OptionMenu(self.root,"random", *choices)
        self.popup_label = Label(self.root, text= "Choose an Active Learning Method:").grid(row = 5, column = 0)
        self.option_menu.grid(row = 6, column = 0)
    
        def classA(event):
            self.img_dict[self.img_list[self.index]]= "0"
            self.getNext()
        def classB(event):
            self.img_dict[self.img_list[self.index]]= "1"
            self.getNext()
        def skipClassEvent(event):
            self.img_dict[self.img_list[self.index]]= "2"
            self.getNext()
        
        self.root.bind(2, classA)
        self.root.bind(1, classB)
        self.root.bind(3,skipClassEvent)
        self.index = 0
        self.img_dict = {}
        # default path if no path is selected yet
        self.path = '/mnt/data1/'
        mainloop()

    #creates dictionary
    def make_pic_dict(self):
        for i in self.img_list:
            if i != self.path+'/labels.pkl':
                self.img_dict[i]= ""

    #loads image on to the screen using a label
    def load_img(self):
        while self.img_dict[self.img_list[self.index]] != "":
            self.index +=1
        im = Image.open(os.path.join(self.path, self.img_list[self.index]))
        photo = ImageTk.PhotoImage(im)
        self.label.config(image=photo, height = 256, width = 256)
        self.label.image = photo

    #gets all the image file names from directory
    #calls for the dictionary to be made
    #loads first image
    def getPaths(self, data_dir):
       image_paths = []
       #exts = ['*.JPG','*.jpg','*.JPEG','*.png','*.PNG']
       #for pattern in exts:
       pattern = '*.JPEG'
       if 1:
           for d, s, fList in os.walk(data_dir):
              for filename in fList:
                 if fnmatch.fnmatch(filename, pattern):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
           print(len(image_paths))
           return image_paths


    def load_img_s(self):
        self.img_list = self.getPaths(self.path)
        if os.path.isfile(self.path+'/labels.pkl'):
            pkl_file = open(self.path+'/labels.pkl', 'rb')
            self.img_dict = pickle.load(pkl_file)
        else:
            self.make_pic_dict()
        self.load_img()

    #allows user to load any path on computer
    def get_path(self):
        self.path = filedialog.askdirectory(initialdir='.')
        self.load_img_s()

    #used for testing contents of the
    def print_list(self):
        if self.img_list == []:
            print(self.img_list)
        else:
            for i in self.img_list:
                print(i)

    #for testing dictionary contents
    def print_dict(self):
        for i in list(self.img_dict.keys()):
            print(i,":",self.img_dict[i])

    #does not allow user to go back past the beginning the list of images
    def getPrev(self):
        index = self.index
        index -=1
        if index > -1:
            self.index = index
            self.load_img()

    #does not allow user to go past the end of the list
    def getNext(self):
        index = self.index
        index +=1
        if index < len(self.img_list):
            self.index = index
            self.load_img()

    #happens each time the user presses quit
    #pushes all data out to text file
    #creates a unique name for each file
    def save(self):
        pkl = open(self.path+'/labels.pkl', 'wb')
        data = pickle.dumps(self.img_dict)
        pkl.write(data)
        pkl.close()
        self.root.destroy()




c = classifier()
