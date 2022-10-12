import librosa
import numpy as np
import math
import common as com
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
import tensorflow as tf
from tensorflow import keras
import numpy as np

class SpecAugment(keras.utils.Sequence):

    def __init__(self, x_in, batch_size, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.datalen = len(x_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

            
    def masking(self, data):
        for idx in range(len(data)):
            vectors_masked = com.spec_augment(data[idx,:,:])
            # data[idx,:,:] = librosa.power_to_db(data[idx,:,:])
            # vectors_masked[idx,:,:] = librosa.power_to_db(vectors_masked[idx,:,:])
            if idx == 0:
                data_masked = np.zeros((len(data), vectors_masked.shape[0], vectors_masked.shape[1]), float)
            if vectors_masked.ndim == 3:
                vectors_masked = np.squeeze(vectors_masked, axis=2)
            data_masked[idx, :, :] = vectors_masked
        return np.swapaxes(data_masked, 2, 1), np.swapaxes(data, 2, 1)
        

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        self.y, self.x = self.masking(self.x)
        self.x = self.x.reshape(1222, self.x.shape[1], self.x.shape[2], 1)
        self.y =  self.y.reshape(1222, self.x.shape[1], self.x.shape[2], 1)
        
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch)
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)