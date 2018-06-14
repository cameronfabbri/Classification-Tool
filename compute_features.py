'''
   File that computes features for a set of images

   ex. python compute_features.py --data_dir=/mnt/images/ --model=vgg19 --model_path=./vgg_19.ckpt

'''

import scipy.misc as misc
import pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import fnmatch
import sys
import os
import load_features as load
#import classifer as c

sys.path.insert(0, 'nets/')

slim = tf.contrib.slim

'''

   Recursively obtains all images in the directory specified

'''
def compute_img_features(model,paths):
   # I only have these because I thought some take in size of (299,299), but maybe not
   if 'inception' in model: height, width, channels = 224, 224, 3
   if 'resnet' in model:    height, width, channels = 224, 224, 3
   if 'vgg' in model:       height, width, channels = 224, 224, 3

   if model == 'inception_resnet_v2': height, width, channels = 299, 299, 3

   x = tf.placeholder(tf.float32, shape=(1, height, width, channels))
   
   # load up model specific stuff
   if model == 'inception_v1':
      import inception_v1
      checkpoint_file = "inception_v1.ckpt"
      arg_scope = inception_v1.inception_v1_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = inception_v1.inception_v1(x, is_training=False, num_classes=1001)
         features = end_points['AvgPool_0a_7x7']
   elif model == 'inception_v2':
      import inception_v2
      checkpoint_file = "inception_v2.ckpt"
      arg_scope = inception_v2.inception_v2_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = inception_v2.inception_v2(x, is_training=False, num_classes=1001)
         features = end_points['AvgPool_1a']
   elif model == 'inception_v3':
      import inception_v3
      arg_scope = inception_v3.inception_v3_arg_scope()
      checkpoint_file = "inception_v3.ckpt"
      with slim.arg_scope(arg_scope):
         logits, end_points = inception_v3.inception_v3(x, is_training=False, num_classes=1001)
         features = end_points['AvgPool_1a']
   elif model == 'inception_resnet_v2':
      import inception_resnet_v2
      arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = inception_resnet_v2.inception_resnet_v2(x, is_training=False, num_classes=1001)
         features = end_points['PreLogitsFlatten']
   elif model == 'resnet_v1_50':
      import resnet_v1
      arg_scope = resnet_v1.resnet_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = resnet_v1.resnet_v1_50(x, is_training=False, num_classes=1000)
         features = end_points['global_pool']
   elif model == 'resnet_v1_101':
      import resnet_v1
      arg_scope = resnet_v1.resnet_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = resnet_v1.resnet_v1_101(x, is_training=False, num_classes=1000)
         features = end_points['global_pool']
   elif model == 'vgg_16':
      import vgg
      arg_scope = vgg.vgg_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = vgg.vgg_16(x, is_training=False)
         features = end_points['vgg_16/fc8']
   elif model == 'vgg_19':
      import vgg
      arg_scope = vgg.vgg_arg_scope()
      with slim.arg_scope(arg_scope):
         logits, end_points = vgg.vgg_19(x, is_training=False)
         features = end_points['vgg_19/fc8']

   sess  = tf.Session()
   saver = tf.train.Saver()
   saver.restore(sess, checkpoint_file)

   feat_dict = {}
   #paths = c.path
   print('Computing features...')
   for path in tqdm(paths):
      image = misc.imread(path)
      image = misc.imresize(image, (height, width))
      image = np.expand_dims(image, 0)
      try: feat = np.squeeze(sess.run(features, feed_dict={x:image}))
      except:
         print('Could not compute feature for',path,'....deleting')
         try: os.remove(path)
         except: continue
         continue
      feat_dict[path] = feat

#try: os.makedirs('features/')
#except: pass
   exp_pkl = open(model+'_features.pkl', 'wb')
   data = pickle.dumps(feat_dict)
   exp_pkl.write(data)
   exp_pkl.close()




