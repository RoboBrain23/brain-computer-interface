from keras.layers import Activation, MaxPooling2D, BatchNormalization,Conv2D, Dense, Dropout, Flatten
from keras.constraints import max_norm
from nn_base import NNBase

class DeepConvNet(NNBase):
    def __init__(self,drop_out = 0.4,kernLength = 256,fs = 128,channels = 2,num_classes = 4):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.dropoutRate = drop_out
        self.kernLength = kernLength
        super().__init__(fs = fs,channels = channels,num_classes = num_classes)
        self.set_input_shape((channels, int(self.window_time*fs), 1))
        self.set_model_name('DeepConvNet')

    def model(self,inputs):
        block1       = Conv2D(25, (1, 5),kernel_constraint = max_norm(2., axis=(0,1,2)))(inputs)
        block1       = Conv2D(25, (inputs.shape[1], 1),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
        block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1       = Activation('elu')(block1)
        block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1       = Dropout(self.dropoutRate)(block1)
      
        block2       = Conv2D(50, (1, 5),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
        block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2       = Activation('elu')(block2)
        block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2       = Dropout(self.dropoutRate)(block2)
        
        block3       = Conv2D(100, (1, 5),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
        block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3       = Activation('elu')(block3)
        block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3       = Dropout(self.dropoutRate)(block3)
        
        block4       = Conv2D(200, (1, 5),
                                    kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
        block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4       = Activation('elu')(block4)
        block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4       = Dropout(self.dropoutRate)(block4)
        
        flatten      = Flatten()(block4)
        
        dense        = Dense(self.num_classes, kernel_constraint = max_norm(0.5))(flatten)
        model      = Activation('softmax')(dense)
        
        return model

