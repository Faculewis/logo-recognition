'''
Created on 23 oct. 2018

@author: sony
'''

from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import functools



class DataImage:    
       
    def __init__(self, path='flickr_logos_27_dataset/flickr_logos_27_dataset_images', 
                 flickr_version=27, image_shape=(160,160), batch_size = 32, test_size = 0.2):
        self.path = path
        self.image_shape = image_shape
        self.flickr_version = flickr_version
        self.batch_size = batch_size
        self.test_size = test_size
        self.switcher = {                    
                    32 : self.flickr_32_paths
                    }
        
        self.flickr_32_paths()
        self.dataset_train, self.dataset_test = self.get_baseline_dataset()
        
    def load_flickr_dataset(self):
        return self.switcher.get(self.flickr_version)()
        
        
#     self.path = /home/sony/Imágenes/FlickrLogos-v2
#     imgages_path = /home/sony/Imágenes/FlickrLogos-v2/classes/jpg
#     mask_path = /home/sony/Imágenes/FlickrLogos-v2/classes/mask
    
    def flickr_32_load(self):
        self.data = []
        self.labels = []
        self.masks = []
#         google,462663740.jpg
        with open(self.path+'/all.txt', 'r') as f:
            for line in f.readlines():
                tokens = line.rstrip('\n').split(',')
                label = tokens[0]
                image_name = self.path+'/classes/jpg/'+label+'/'+tokens[1]
                mask_name = self.path+'classes/masks/'+label+'/'+tokens[1]+'.mask.0.png'
                img = image.load_img(image_name, target_size=self.image_shape)
                img = image.img_to_array(img)
                img /= 255.
                if label != 'no-logo':
                    mask = image.load_img(mask_name, target_size=self.image_shape)
                    mask = image.img_to_array(mask)
                    self.masks.append(mask)
                self.data.append(img)
                self.labels.append(label)
                
        
        return (self.data, self.labels, self.masks)
    
    def flickr_32_paths(self):
        self.data = []
        self.labels = []
        self.masks = []
    #         google,462663740.jpg
        with open(self.path+'/all.txt', 'r') as f:
            for line in f.readlines():
                tokens = line.rstrip('\n').split(',')
                label = tokens[0]
                
                image_name = self.path+'classes/jpg/'+label+'/'+tokens[1]                

                if label != 'no-logo':
                    mask_name = self.path+'classes/masks/'+label+'/'+tokens[1]+'.mask.merged.png'
                else:
                    mask_name = self.path+'classes/masks/no-logo.png'
                    
                self.data.append(image_name)
                self.masks.append(mask_name)
                self.labels.append(label)

#         data = np.asarray(data)
#         masks = np.asarray(masks)
#         labels = np.asarray(labels)
        return (self.data, self.labels, self.masks)
    
#     def _process_pathnames(self, fname, label_path):
#   
#         img_str = tf.read_file(fname)
#         img = tf.image.decode_jpeg(img_str, channels=3)
#         
#         label_img_str = tf.read_file(label_path)
#         # These are gif images so they return as (num_frames, h, w, c)
#         label_img = tf.image.decode_gif(label_img_str)[0]
#         # The label image should only have values of 1 or 0, indicating pixel wise
#         # object (car) or not (background). We take the first channel only. 
#         label_img = label_img[:, :, 0]
#         label_img = tf.expand_dims(label_img, axis=-1)
#         return img, label_img
    
    def process_pathnames(self, fname, mask_path):
        img_str = tf.read_file(fname)
        img = tf.image.decode_jpeg(img_str, channels=3)
        
        mask_img_str = tf.read_file(mask_path)
        mask_img = tf.image.decode_gif(mask_img_str)[0]
        
        mask_img = mask_img[:, :, 0]
        mask_img = tf.expand_dims(mask_img, axis=-1)
        return img, mask_img
    
#     def _augment(self,
#              img,
#              label_img,
#              resize=None,  # Resize the image to some size e.g. [256, 256]
#              scale=1,  # Scale image e.g. 1 / 255.
#              hue_delta=0,  # Adjust the hue of an RGB image by random factor
#              ):  # Randomly translate the image vertically 
# #         if resize is not None:
# #             # Resize both images
# #             label_img = tf.image.resize_images(label_img, resize)
# #             img = tf.image.resize_images(img, resize)
# #         
# #         if hue_delta:
# #             img = tf.image.random_hue(img, hue_delta)
#         
#         label_img = tf.to_float(label_img) * scale
#         img = tf.to_float(img) * scale 
#         return img, label_img
    
    def augment (self, img, mask, resize=(160,160), scale = 1.):
            mask = tf.image.resize_images(mask, resize)            
            mask = tf.to_float(mask) * scale
            
            img = tf.image.resize_images(img, resize)
            img = tf.to_float(img) * scale            
            return img, mask
    

    def get_baseline_dataset(self,
                         threads=8, 
                         batch_size=None,
                         shuffle = True):
          
        if batch_size is None:
            batch_size = self.batch_size
                             
        assert len(self.data) == len(self.masks)
        
        (X_train, X_test, y_train, y_test) = train_test_split(self.data,self.masks, test_size=self.test_size, random_state=42)
        
        self.num_train = len(X_train)
        self.num_test = len(X_test)
        
        # Create a dataset from the filenames and labels
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        # Map our preprocessing function to every element in our dataset, taking
        # advantage of multithreading               
        procces = functools.partial(self.process_pathnames)
        
        dataset_train = dataset_train.map(procces, num_parallel_calls=threads)
        dataset_test = dataset_test.map(procces, num_parallel_calls=threads)

        tr_cfg = {
        'resize': [160, 160],
        'scale': 1 / 255.,
        }
        augment = functools.partial(self.augment, **tr_cfg)
   
        dataset_train = dataset_train.map(augment, num_parallel_calls=threads)
        dataset_test = dataset_test.map(augment, num_parallel_calls=threads)
        
        if shuffle:
            dataset_train = dataset_train.shuffle(self.num_train)
            dataset_test = dataset_test.shuffle(self.num_test)
        
        # It's necessary to repeat our data for all epochs 
        dataset_train = dataset_train.repeat().batch(batch_size)
        dataset_test = dataset_test.repeat().batch(batch_size)
        return dataset_train, dataset_test
    
    