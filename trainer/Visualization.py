'''
Created on 15 dic. 2018

@author: sony
'''

from src.DataImage import DataImage
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)
import matplotlib.image as mpimg 
import pandas as pd
import numpy as np
from PIL import Image

class Visual(object):
    '''
    classdocs
    '''


    def __init__(self, data, model=None):
        '''
        Constructor
        '''
        self.data = data
        self.model = model
        
    def prueba(self):
        y_pathname = '/home/sony/Imágenes/FlickrLogos-v2/classes/masks/adidas/3068575660.jpg.mask.merged.png'
        x_pathname = '/home/sony/Imágenes/FlickrLogos-v2/classes/jpg/adidas/3068575660.jpg'
        
        plt.figure(figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(mpimg.imread(x_pathname))
        plt.title("Original Image")
          
        example_labels = Image.open(y_pathname)      
        plt.subplot(1, 2, 2)
        plt.imshow(example_labels)
        plt.title("Masked Image")  
        
        plt.suptitle("Examples of Images and their Masks")
        plt.show()
        
    def  img_and_mask_pipeline(self):
        
        temp_ds = self.data.dataset
        # Let's examine some of these augmented images
        data_aug_iter = temp_ds.make_one_shot_iterator()
        next_element = data_aug_iter.get_next()
        with tf.Session() as sess:
            batch_of_imgs, label = sess.run(next_element)
            
            # Running next element in our graph will produce a batch of images
            plt.figure(figsize=(10, 10))
            img = batch_of_imgs[0]
            
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            
            plt.subplot(1, 2, 2)
            plt.imshow(label[0, :, :, 0])
            plt.show()
            
    
    def img_and_mask(self, display_num=5):
        x_train_filenames = self.data.data
        y_train_filenames = self.data.masks
        num_train_examples = len(self.data.data)
        
        r_choices = np.random.choice(num_train_examples, display_num)
        
        plt.figure(figsize=(10, 15))
        for i in range(0, display_num * 2, 2):
            img_num = r_choices[i // 2]
            x_pathname = x_train_filenames[img_num]
            y_pathname = y_train_filenames[img_num]
              
            plt.subplot(display_num, 2, i + 1)
            plt.imshow(mpimg.imread(x_pathname))
            plt.title("Original Image")
              
            example_labels = Image.open(y_pathname)
            label_vals = np.unique(example_labels)
              
            plt.subplot(display_num, 2, i + 2)
            plt.imshow(example_labels)
            plt.title("Masked Image")  
          
        plt.suptitle("Examples of Images and their Masks")
        plt.show()
            
            
            