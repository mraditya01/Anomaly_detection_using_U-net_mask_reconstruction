########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.initializers import orthogonal
from keras import activations

########################################################################
# keras model
########################################################################


def Conv2DLayer(x, filters, kernel, strides, padding, block_id):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                       name=prefix+'conv')(x)
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    # x = layers.Dropout(0.5, name=prefix+'drop')((x))
    return x


def Conv2DLayer2(x, filters, kernel, strides, padding, block_id):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                       name=prefix+'conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    # x = layers.Dropout(0.5, name=prefix+'drop')((x))
    return x

def Transpose_Conv2D(x, filters, kernel, strides, padding, block_id):
    prefix = f'block_{block_id}_'
    x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                                name=prefix+'de-conv')(x)
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')((x))
    return x

def Transpose_Conv2D2(x, filters, kernel, strides, padding, block_id):
  prefix = f'block_{block_id}_'
  x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                              name=prefix+'de-conv')(x)
  x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
  x = layers.LeakyReLU(name=prefix+'lrelu')(x)
#   x = layers.Dropout(0.2, name=prefix+'drop')((x))
  return x


#########################################################################

def get_model_4(input_shape):
    inputs = layers.Input(shape=input_shape)
    conv1 = Conv2DLayer(inputs, 16, (3, 3), strides=(1, 1), padding='same', block_id=1)
    conv1 = Conv2DLayer(conv1, 16, (3, 3), strides=(1, 1), padding='same', block_id=2)
    maxpool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool1')(conv1)
    conv2 = Conv2DLayer(maxpool1, 32, (3, 3), strides=(1, 1), padding='same', block_id=3)
    conv2 = Conv2DLayer(conv2, 32, (3, 3), strides=(1, 1), padding='same', block_id=4)
    maxpool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool2')(conv2)
    conv3 = Conv2DLayer(maxpool2, 64, (3, 3), strides=(1, 1), padding='same', block_id=5)
    conv3 = Conv2DLayer(conv3, 64, (3, 3), strides=(1, 1), padding='same', block_id=6)
    maxpool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool3')(conv3)
    conv4 = Conv2DLayer(maxpool3, 128, (3, 3), strides=(1, 1), padding='same', block_id=7)
    conv4 = Conv2DLayer(conv4, 128, (3, 3), strides=(1, 1), padding='same', block_id=8)
    maxpool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool4')(conv4)
    conv5 = Conv2DLayer(maxpool4, 256, (3, 3), strides=(1, 1), padding='same', block_id=9)
    conv5 = Conv2DLayer(conv5, 256, (3, 3), strides=(1, 1), padding='same', block_id=10)

    deconv1 = layers.UpSampling2D(size=(2,2), name='upsampling1')(conv5)
    
    skip1 = layers.concatenate([deconv1, conv4], name='skip1')
    conv7 = Conv2DLayer(skip1, 128, (3, 3), strides=(1, 1), padding='same', block_id=11)
    conv7 = Conv2DLayer(conv7, 128, (3, 3), strides=(1, 1), padding='same', block_id=12)
    deconv2 = layers.UpSampling2D(size=(2,2), name='upsampling2')(conv7)
    
    skip2 = layers.concatenate([deconv2, conv3], name='skip2')
    conv8 = Conv2DLayer(skip2, 64, (3, 3), strides=(1, 1), padding='same', block_id=13)
    conv8 = Conv2DLayer(conv8, 64, (3, 3), strides=(1, 1), padding='same', block_id=14)
    deconv3 = layers.UpSampling2D(size=(2,2), name='upsampling3')(conv8)
    
    skip3 = layers.concatenate([deconv3, conv2], name='skip3')
    conv9 = Conv2DLayer(skip3, 32, (3, 3), strides=(1, 1), padding='same', block_id=15)
    conv9 = Conv2DLayer(conv9, 32, (3, 3), strides=(1, 1), padding='same', block_id=16)
    deconv4 = layers.UpSampling2D(size=(2,2), name='upsampling4')(conv9)
    
    skip4 = layers.concatenate([deconv4, conv1], name='skip4')
    conv10 = Conv2DLayer(skip4, 16, (3, 3), strides=(1, 1), padding='same', block_id=17)
    conv10 = Conv2DLayer(conv10, 16, (3, 3), strides=(1, 1), padding='same', block_id=18)
    conv10 = Conv2DLayer2(conv10, 1, (3, 3), strides=(1, 1), padding='same', block_id=19)

    # deconv5 = Transpose_Conv2D2(conv10, 1, (3, 3), strides=(1, 1), padding='same', block_id=14)
    

    model = Model(inputs=inputs, outputs=conv10)    
    return model

def get_model_3(input_shape):
    inputs = layers.Input(shape=input_shape)
    conv1 = Conv2DLayer(inputs, 16, (3, 3), strides=(1, 1), padding='same', block_id=1)
    maxpool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool1')(conv1)
    conv2 = Conv2DLayer(maxpool1, 32, (3, 3), strides=(1, 1), padding='same', block_id=2)
    maxpool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool2')(conv2)
    conv3 = Conv2DLayer(maxpool2, 64, (3, 3), strides=(1, 1), padding='same', block_id=3)
    maxpool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool3')(conv3)
    conv4 = Conv2DLayer(maxpool3, 128, (3, 3), strides=(1, 1), padding='same', block_id=4)
    maxpool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool4')(conv4)
    conv5 = Conv2DLayer(maxpool4, 256, (3, 3), strides=(1, 1), padding='same', block_id=5)
   

    deconv1 = Transpose_Conv2D(conv5, 128, (3, 3), strides=(2, 2), padding='same', block_id=6)
    
    skip1 = layers.concatenate([deconv1, conv4], name='skip1')
    conv7 = Conv2DLayer(skip1, 128, (3, 3), strides=(1, 1), padding='same', block_id=7)
    deconv2 = Transpose_Conv2D(conv7, 64, (3, 3), strides=(2, 2), padding='same', block_id=8)
    
    skip2 = layers.concatenate([deconv2, conv3], name='skip2')
    conv8 = Conv2DLayer(skip2, 64, (3, 3), strides=(1, 1), padding='same', block_id=9)
    deconv3 = Transpose_Conv2D2(conv8, 32, (3, 3), strides=(2, 2), padding='same', block_id=10)
    
    skip3 = layers.concatenate([deconv3, conv2], name='skip3')
    conv9 = Conv2DLayer(skip3, 32, (3, 3), strides=(1, 1), padding='same', block_id=11)
    deconv4 = Transpose_Conv2D2(conv9, 16, (3, 3), strides=(2, 2), padding='same', block_id=12)
    
    skip4 = layers.concatenate([deconv4, conv1], name='skip4')  
    conv10 = Conv2DLayer2(skip4, 1, (3, 3), strides=(1, 1), padding='same', block_id=13)

    # deconv5 = Transpose_Conv2D2(conv10, 1, (3, 3), strides=(1, 1), padding='same', block_id=14)
    

    model = Model(inputs=inputs, outputs=conv10)    
    return model

def get_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    conv1 = Conv2DLayer(inputs, 16, (5, 5), strides=(2, 2), padding='same', block_id=1)
    conv2 = Conv2DLayer(conv1, 32, (5, 5), strides=(2, 2), padding='same', block_id=2)
    conv3 = Conv2DLayer(conv2, 64, (5, 5), strides=(2, 2), padding='same', block_id=3)
    conv4 = Conv2DLayer(conv3, 128, (5, 5), strides=(2, 2), padding='same', block_id=4)
    conv5 = Conv2DLayer(conv4, 256, (5, 5), strides=(2, 2), padding='same', block_id=5)
    conv6 = Conv2DLayer(conv5, 512, (5, 5), strides=(2, 2), padding='same', block_id=6)
   

    deconv1 = Transpose_Conv2D(conv6, 256, (5, 5), strides=(2, 2), padding='same', block_id=7)
    
    skip1 = layers.concatenate([deconv1, conv5], name='skip1')
    deconv2 = Transpose_Conv2D(skip1, 128, (5, 5), strides=(2, 2), padding='same', block_id=8)
    
    skip2 = layers.concatenate([deconv2, conv4], name='skip2')
    deconv3 = Transpose_Conv2D2(skip2, 64, (5, 5), strides=(2, 2), padding='same', block_id=9)
    
    skip3 = layers.concatenate([deconv3, conv3], name='skip3')
    deconv4 = Transpose_Conv2D2(skip3, 32, (5, 5), strides=(2, 2), padding='same', block_id=10)
    
    skip4 = layers.concatenate([deconv4, conv2], name='skip4')    
    deconv5 = Transpose_Conv2D2(skip4, 16, (5, 5), strides=(2, 2), padding='same', block_id=11)
    
    skip5 = layers.concatenate([deconv5, conv1], name='skip5')
    deconv6 = Transpose_Conv2D2(skip5, 1, (5, 5), strides=(2, 2), padding='same', block_id=12)
    

    model = Model(inputs=inputs, outputs=deconv6)    
    return model

def get_model_2(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    conv1 = Conv2DLayer(inputs, 16, (5, 5), strides=(2, 2), padding='same', block_id=1)
    conv2 = Conv2DLayer(conv1, 32, (5, 5), strides=(2, 2), padding='same', block_id=2)
    conv3 = Conv2DLayer(conv2, 64, (5, 5), strides=(2, 2), padding='same', block_id=3)
    conv4 = Conv2DLayer(conv3, 128, (5, 5), strides=(2, 2), padding='same', block_id=4)
    conv5 = Conv2DLayer(conv4, 256, (5, 5), strides=(2, 2), padding='same', block_id=5)
    conv6 = Conv2DLayer(conv5, 512, (5, 5), strides=(2, 2), padding='same', block_id=6)
   

    deconv1 = Transpose_Conv2D(conv6, 256, (5, 5), strides=(2, 2), padding='same', block_id=7)
    
    skip1 = layers.concatenate([deconv1, conv5], name='skip1')
    conv7 = Conv2DLayer(skip1, 256, (5, 5), strides=(2, 2), padding='same', block_id=8)
    deconv2 = Transpose_Conv2D(conv7, 128, (5, 5), strides=(4, 4), padding='same', block_id=9)
    
    skip2 = layers.concatenate([deconv2, conv4], name='skip2')
    conv8 = Conv2DLayer(skip2, 128, (5, 5), strides=(2, 2), padding='same', block_id=10)
    deconv3 = Transpose_Conv2D2(conv8, 64, (5, 5), strides=(4, 4), padding='same', block_id=11)
    
    skip3 = layers.concatenate([deconv3, conv3], name='skip3')
    conv9 = Conv2DLayer(skip3, 64, (5, 5), strides=(2, 2), padding='same', block_id=12)
    deconv4 = Transpose_Conv2D2(conv9, 32, (5, 5), strides=(4, 4), padding='same', block_id=13)
    
    skip4 = layers.concatenate([deconv4, conv2], name='skip4')  
    conv10 = Conv2DLayer(skip4, 32, (5, 5), strides=(2, 2), padding='same', block_id=14)  
    deconv5 = Transpose_Conv2D2(conv10, 16, (5, 5), strides=(4, 4), padding='same', block_id=15)
    
    skip5 = layers.concatenate([deconv5, conv1], name='skip5')
    conv11 = Conv2DLayer(skip5, 16, (5, 5), strides=(2, 2), padding='same', block_id=16)
    deconv6 = Transpose_Conv2D2(conv11, 1, (5, 5), strides=(4, 4), padding='same', block_id=17)
    

    model = Model(inputs=inputs, outputs=deconv6)    
    return model


# def get_model(input_shape):
#     inputs = layers.Input(shape=input_shape)
    
#     # 256 x 256
#     conv1 = Conv2DLayer(inputs, 64, 3, strides=1, padding='same', block_id=1)
#     conv2 = Conv2DLayer(conv1, 64, 3, strides=2, padding='same', block_id=2)
    
#     # 128 x 128
#     conv3 = Conv2DLayer(conv2, 128, 5, strides=2, padding='same', block_id=3)
    
#     # 64 x 64
#     conv4 = Conv2DLayer(conv3, 128, 3, strides=1, padding='same', block_id=4)
#     conv5 = Conv2DLayer(conv4, 256, 5, strides=2, padding='same', block_id=5)
    
#     # 32 x 32
#     conv6 = Conv2DLayer(conv5, 512, 3, strides=2, padding='same', block_id=6)
    
#     # 16 x 16
#     deconv1 = Transpose_Conv2D(conv6, 512, 3, strides=2, padding='same', block_id=7)
    
#     # 32 x 32
#     skip1 = layers.concatenate([deconv1, conv5], name='skip1')
#     conv7 = Conv2DLayer(skip1, 256, 3, strides=1, padding='same', block_id=8)
#     deconv2 = Transpose_Conv2D(conv7, 128, 3, strides=2, padding='same', block_id=9)
    
#     # 64 x 64
#     skip2 = layers.concatenate([deconv2, conv3], name='skip2')
#     conv8 = Conv2DLayer(skip2, 128, 5, strides=1, padding='same', block_id=10)
#     deconv3 = Transpose_Conv2D(conv8, 64, 3, strides=2, padding='same', block_id=11)
    
#     # 128 x 128
#     skip3 = layers.concatenate([deconv3, conv2], name='skip3')
#     conv9 = Conv2DLayer(skip3, 64, 5, strides=1, padding='same', block_id=12)
#     deconv4 = Transpose_Conv2D(conv9, 32, 3, strides=2, padding='same', block_id=13)
    
#     # 256 x 256
#     skip4 = layers.concatenate([deconv4, conv1])
#     conv9 = Conv2DLayer(skip4, 32, 5, strides=1, padding='same', block_id=14)

#     model = Model(inputs=inputs, outputs=conv9)
#     return model

def load_model(file_path):
    return keras.models.load_model(file_path, compile=False)