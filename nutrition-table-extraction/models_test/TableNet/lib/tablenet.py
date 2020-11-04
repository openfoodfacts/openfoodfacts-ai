import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D

class TableNet:
    @staticmethod
    def build_table_decoder(inputs, pool3, pool4):
        x = Conv2D(512, (1, 1), activation = 'relu', name='conv7_table')(inputs)
        x = UpSampling2D(size=(2, 2))(x)
        concatenated = Concatenate()([x, pool4])
        # concatenated = concatenate([x, pool4])
        x = UpSampling2D(size=(2,2))(concatenated)
        
        concatenated = Concatenate()([x, pool3])
        x = UpSampling2D(size=(2,2))(concatenated)
        x = UpSampling2D(size=(2,2))(x)
        last = tf.keras.layers.Conv2DTranspose(
          3, 3, strides=2,
          padding='same', name='table_output') 
        
        x = last(x)
        return x
    @staticmethod
    def build_column_decoder(inputs, pool3, pool4):
      
        x = Conv2D(512, (1, 1), activation = 'relu', name='block7_conv1_column')(inputs)
        x = Dropout(0.8, name='block7_dropout_column')(x)
        x = Conv2D(512, (1, 1), activation = 'relu', name='block8_conv1_column')(x)
        x = UpSampling2D(size=(2, 2))(x)
        concatenated = Concatenate()([x, pool4])
        # concatenated = concatenate([x, pool4])
        x = UpSampling2D(size=(2,2))(concatenated)
        
        concatenated = Concatenate()([x, pool3])
        x = UpSampling2D(size=(2,2))(concatenated)
        x = UpSampling2D(size=(2,2))(x)
        last = tf.keras.layers.Conv2DTranspose(
          3, 3, strides=2,
          padding='same', name='column_output') 
        
        x = last(x)
        return x  
    @staticmethod
    def vgg_base(inputs):
        base_model = tf.keras.applications.vgg19.VGG19(
            input_shape=[256, 256, 3],
            include_top=False, weights='imagenet')
        
        layer_names = ['block3_pool', 'block4_pool', 'block5_pool']
        layers = [base_model.get_layer(name).output for name in layer_names]
        pool_layers_model = Model(inputs=base_model.input, outputs=layers, name='VGG-19')
        pool_layers_model.trainable = False
        return pool_layers_model(inputs)
    
    @staticmethod
    def build():
        inputShape = (256, 256, 3)
        inputs = Input(shape=inputShape, name='input')
        pool_layers = TableNet.vgg_base(inputs)
        x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv1')(pool_layers[2])
        x = Dropout(0.8, name='block6_dropout1')(x)
        x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv2')(x)
        x = Dropout(0.8, name = 'block6_dropout2')(x)
        
        table_mask = TableNet.build_table_decoder(x, pool_layers[0], pool_layers[1])
        column_mask = TableNet.build_column_decoder(x, pool_layers[0], pool_layers[1])
        model = Model(inputs=inputs,outputs=[table_mask, column_mask],name="tablenet")
        
        return model