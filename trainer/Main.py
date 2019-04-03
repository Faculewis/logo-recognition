'''
Created on 12 dic. 2018

@author: sony
'''
from src.DataImage import DataImage
from src.ModelMaker import ModelMaker
from src.CallBackMaker import CallBackMaker
import numpy as np
import src.LossFunctionsS as l
from src.Visualization import Visual
from tensorflow.python.keras import backend as K
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


folder_path = '/home/sony/Imágenes/FlickrLogos-v2/'
flickr_version = 32
model_type = 'UNet' 

image_shape = (160,160)
input_shape = (160,160,3)
epochs = 100
batch_size = 36

if __name__ == '__main__':
        
        K.clear_session()
        print('Starting Images, Labels and masks load...')
        data = DataImage(folder_path, flickr_version, image_shape, batch_size)    
#         dataset = data.dataset    
        print("Visualization...")
        v = Visual(data)
        
#         v.prueba()
#         v.img_and_mask(5)
#         print("End Visualization...") 
#         print('Cantidad de elementos del dataset')   
#         print('len(data) '+len(data.data).__str__())
#         print('len(mask) '+len(data.masks).__str__())
#         
#         v.img_and_mask_pipeline()
        
        print('Algunas rutas...') 
        print(data.data[:10])
        print(data.masks[:10])
         
        num_train_examples = data.num_train
        num_val_examples = data.num_test
        
        print('End DataSet load...')
        
        print('Making model...')
          
        modelMaker = ModelMaker(input_shape, 'same')
        model = modelMaker.model
        print('End making model...')
           
        print(model.summary()) 
        
        print('Making Callbacks...')
        callbackMaker = CallBackMaker(model_type=model_type, flickr_version=flickr_version)
        print('End making Callbacks')
        optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
           
        model.compile(optimizer='adam',
                      loss= l.bce_dice_loss, metrics=[l.dice_loss, 'acc'])
        
        call_backs = [ModelCheckpoint('/home/sony/eclipse-workspace/Logo-Image-Segmentation/callbacks/checkpoints/'+model_type+'_'+str(flickr_version)+'.h5',
                                               monitor='val_dice_loss', verbose=0, 
                                               save_best_only=True, save_weights_only=False, mode='auto', period=1),
                      EarlyStopping(monitor='val_dice_loss', min_delta=0, patience=8, verbose=0, mode='auto'),
#                       ReduceLROnPlateau(monitor='dice_loss', factor=0.01, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
#                     TensorBoard(log_dir='/home/sony/eclipse-workspace/Logo-Image-Segmentation/callbacks/tensorboard/', batch_size=batch_size, histogram_freq=0, update_freq='epoch', write_graph=True, write_grads=False, write_images=True)
                        ]
        
        
   
        history = model.fit(data.dataset_train,
                            steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),                           
                            epochs=epochs,
                            validation_data = data.dataset_test,
                            validation_steps = int(np.ceil(num_val_examples / float(batch_size))), 
                            callbacks = call_backs)
           
        
pass