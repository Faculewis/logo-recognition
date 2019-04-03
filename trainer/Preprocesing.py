'''
Created on 14 dic. 2018

@author: sony
'''
import tensorflow as tf

class Preprocessing():
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
    def process_pathnames(self, fname, mask_path):
        img_str = tf.read_file(fname)
        img = tf.image.decode_jpeg(img_str, channels=3)
        
        mask_img_str = tf.read_file(mask_path)
        mask_img = tf.image.decode_png(mask_img_str)[0]
        
        mask_img = mask_img[:,:,0]
        mask_img = tf.expand_dims(mask_img, axis=-1)
        return img, mask_img
    
    def get_baseline_dataset(self,filenames, 
                         masks,                         
                         threads=4, 
                         batch_size=25):           
        num_x = len(filenames)
        # Create a dataset from the filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, masks))
        # Map our preprocessing function to every element in our dataset, taking
        # advantage of multithreading
        procces = self.process_pathnames()
        dataset = dataset.map(procces, num_parallel_calls=threads)        
        
        dataset = dataset.map(None, num_parallel_calls=threads)        
        
        # It's necessary to repeat our data for all epochs 
        dataset = dataset.repeat().batch(batch_size)
        return dataset