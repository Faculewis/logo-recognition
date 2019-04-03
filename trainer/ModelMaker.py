'''
Created on 13 dic. 2018

@author: sony
'''
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from matplotlib import axis
from numpy.core.defchararray import center



class ModelMaker(object):
    '''
    classdocs
    '''


    def __init__(self, input_shape,padding_type):
        '''
        Constructor
        '''
        self.padding_type = padding_type
        
        inputs = layers.Input(shape=input_shape)
        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)
        
        center = self.conv_block(encoder4_pool, 1024)
        
        decoder4 = self.decoder_block(center, encoder4, 512)
        decoder3 = self.decoder_block(decoder4, encoder3, 256)
        decoder2 = self.decoder_block(decoder3, encoder2, 128)
        decoder1 = self.decoder_block(decoder2, encoder1, 64)
        decoder0 = self.decoder_block(decoder1, encoder0, 32)
        
        outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(decoder0)
        
        self.model = models.Model(inputs = [inputs], outputs = [outputs])
        
        
        
    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3,3), padding=self.padding_type)(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3,3), padding=self.padding_type)(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder
    
    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPool2D((2,2), strides=(2,2))(encoder)
        return encoder_pool, encoder
    
    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding=self.padding_type)(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3,3), padding=self.padding_type)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3,3), padding=self.padding_type)(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder
    
    
    
    
        
      