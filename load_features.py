'''

   Base file for using the features that were computed and stored in the pickle file.
   This is tensorflow agnostic

   Ex. python load_features.py inception_v1_features.pkl

'''

import numpy as np
import sys
import pickle


def load_img_features(model,init_dir):
    with open(init_dir+'/'+model+'_features.pkl', 'rb') as pickle_file:
        features = pickle.load(pickle_file)
    #for image, feature in features.items():
        #print(image, ':', feature)
    return features

