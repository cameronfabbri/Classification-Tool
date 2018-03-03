from tkinter import *
from tkinter import filedialog
import os
from PIL import Image,ImageTk
from time import gmtime, strftime  #used to make unique file names to write out to 

import _pickle as pickle
from sklearn import svm

'''
   need: I need to implement a way to start
   from an existing textfile of written data
   that way one could technically start from
   where they left off
'''

'''
   TODO:

   - find images recursively
   - drop down box selecting active learning method (current would be 'random')
   - Say which class is which for buttons 1 and 2
   - load previous pickle file
   - max image size so the image doesn't change window size
   - 
'''

# this is how you load a pickle file. 'a' is then the dictionary that we saved
# pkl_file = open(sys.argv[1], 'rb')
# a = pickle.load(pkl_file)


class classifier():

    # gets all images in subdirectories
    def getPaths(data_dir):
       exts = ['png', 'PNG','jpg', 'JPG', 'JPEG']
       for e in exts:
           image_paths = []
           for d, s, fList in os.walk(data_dir):
               for filename in fList:
                   if fnmatch.fnmatch(filename, pattern):
                       fname_ = os.path.join(d,filename)
                       image_paths.append(fname_)
       return image_paths


    #sers up all buttons and binds keys for classification
    def __init__(self, root=None):
        self.img_list = []
        self.root = Tk()
        self.window = Frame(self.root)
        self.root.title("Picture Classifier")
        self.root.geometry('500x300')
        self.quit = Button(self.root,text = "Quit", command = self.save)
        self.quit.grid(column= 0,row = 0, sticky = E)
        self.next = Button(self.root,text = "Next", command = self.getNext)
        self.next.grid(column= 0, row = 1, sticky = E)
        self.prev = Button(self.root,text = "Previous", command = self.getPrev)
        self.prev.grid(column = 0, row = 2, sticky = E)
        self.load = Button(self.root,text = "Load", command = self.get_path)
        self.load.grid(column = 0, row = 3, sticky = E)
        self.label = Label(text= "No Image Loaded")
        self.label.grid(row = 0, column = 1, columnspan = 10)

        def distorted(event):
            self.img_dict[self.img_list[self.index]]= 0
            self.getNext()
        def normal(event):
            self.img_dict[self.img_list[self.index]]= 1
            self.getNext()
        self.root.bind(2, distorted) # I should make it so it moves to the next image
        self.root.bind(1, normal)  # after user presses button
        self.index = 0
        self.img_dict = {}
        self.path = ""
        mainloop()

    #creates dictionary
    def make_pic_dict(self):
        for i in self.img_list:
            self.img_dict[i]= ""

    #loads image on to the screen using a label
    def load_img(self):
        im = Image.open(os.path.join(self.path, self.img_list[self.index]))
        photo = ImageTk.PhotoImage(im)
        self.label.config(image=photo)
        self.label.image = photo


    #gets all the image file names from directory
    #calls for the dictionary to be made
    #loads first image
    def load_img_s(self):
        self.img_list = os.listdir(self.path)
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
