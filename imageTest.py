import fnmatch
from sklearn.svm import SVC
import sklearn
import scipy.misc as misc
import os
import numpy as np
from tqdm import tqdm
import cv2

def getPaths(data_dir):
   paths = []
   pattern = '*.JPEG'
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            paths.append(fname_)
   return paths

if __name__ == '__main__':

   paths = getPaths('temp/')[:10]

   unlabeledImg   = []
   allLabel = []

   clf = SVC()

   for p in paths:
      image = cv2.imread(p)
      image = cv2.resize(image, (256,256))
      unlabeledImg.append(image)

   labeledImgs = []

   current = 0
   cv2.namedWindow("Binary classifier")
   while(1):
      image = unlabeledImg[current]
      cv2.imshow("Binary classifier", image)
      key = cv2.waitKey(0) & 0xFF
      if key == ord('a'):
         print 'labelA'
         allLabel.append(0)
         labeledImgs.append(image.flatten())
      elif key == ord('o'):
         print 'labelB'
         allLabel.append(1)
         labeledImgs.append(image.flatten())
      elif key == ord('q'):
         cv2.destroyAllWindows()
         exit()
      if current == len(paths)-1:
         break

      if 0 in allLabel and 1 in allLabel:
         clf.fit(np.asarray(labeledImgs),np.asarray(allLabel))
         print clf.decision_function(np.asarray(labeledImgs))

      current += 1
   cv2.destroyAllWindows()

