'''
Created on 24 oct. 2018

@author: sony
'''
import os
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CallBackMaker:
    '''
    classdocs
    '''

    def __init__(self, model_type, flickr_version, monitor = 'dice_loss',
                 early_stopping = False, reduce_lr_on_plateau = True,
                 tensorboard = False, model_checkpoint = True,
                 checkpoint_path = '/home/sony/eclipse-workspace/Logo-Image-Segmentation/callbacks/checkpoints/',
                 tensorboard_path = '/home/sony/eclipse-workspace/Logo-Image-Segmentation/callbacks/tensorboard/'):
        '''
        Constructor
        '''
        self.monitor = monitor
        self.tensorboard_path = tensorboard_path
        self.checkpoint_path = checkpoint_path
        self.call_backs = []
        
        early_stopping = EarlyStopping(monitor='dice_loss', min_delta=0, patience=8, verbose=0, mode='auto')
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='dice_loss', factor=0.01, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        tensor_board = TensorBoard(log_dir=self.tensorboard_path, histogram_freq=0, update_freq='epoch', write_graph=True, write_grads=True, write_images=True)
        model_checkpoint = ModelCheckpoint(self.checkpoint_path+model_type+'_'+str(flickr_version)+'.h5',
                                           monitor='val_loss', verbose=0, 
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

        self.call_backs = [            
            early_stopping,
            reduce_lr_on_plateau, 
            model_checkpoint, 
            tensor_board
        ]
        
        
        pass
    