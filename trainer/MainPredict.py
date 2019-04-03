'''
Created on 17 dic. 2018

@author: sony
'''

from src.DataImage import DataImage
from src.ModelMaker import ModelMaker
from src.CallBackMaker import CallBackMaker
import numpy as np
import src.LossFunctionsS as l
from src.Visualization import Visual
from tensorflow.python.keras import backend as K
from keras import optimizers, models
import tensorflow as tf

folder_path = '/home/sony/Imágenes/FlickrLogos-v2/'
flickr_version = 32
model_type = 'UNet' 

image_shape = (160,160)
input_shape = (160,160,3)
epochs = 100
batch_size = 36

model_checkpoint_path = '/home/sony/eclipse-workspace/Logo-Image-Segmentation/callbacks/checkpoints/'+model_type+'_'+str(flickr_version)+'.h5'

if __name__ == '__main__':
    
    model = models.load_model(model_checkpoint_path, custom_objects={'bce_dice_loss': l.bce_dice_loss,
                                                           'dice_loss': l.dice_loss})
    
    
    
    pass