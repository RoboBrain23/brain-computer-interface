from keras.layers import Activation, AveragePooling2D, BatchNormalization,Conv2D, Dense, Dropout, Flatten
from keras.constraints import max_norm
from nn_base import NNBase
from utils import log,square

class ShallowConvNet(NNBase):
    def __init__(self,drop_out = 0.4,kernLength = 256,fs = 128,channels = 2,num_classes = 4):
        super().__init__(fs = fs,channels = channels,num_classes = num_classes)
        self.dropoutRate = drop_out
        self.kernLength = kernLength
        self.set_input_shape((channels, int(self.window_time*fs), 1))
        self.set_model_name('ShallowConvNet')

    def model(self,inputs):
        block1       = Conv2D(40, (1, 13),kernel_constraint = max_norm(2., axis=(0,1,2)))(inputs)
        block1       = Conv2D(40, (inputs.shape[1], 1), use_bias=False,kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
        block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1       = Activation(square)(block1)
        block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
        block1       = Activation(log)(block1)
        flatten      = Flatten()(block1)
        block1       = Dropout(self.dropoutRate)(block1)
        dense        = Dense(self.num_classes, kernel_constraint = max_norm(0.5))(flatten)
        model        = Activation('softmax')(dense)
        
        return model

