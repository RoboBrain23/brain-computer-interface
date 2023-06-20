from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, DepthwiseConv2D, Dropout, Flatten,
                          SeparableConv2D, SpatialDropout2D)
from keras.constraints import max_norm
from nn_base import NNBase


class EEGNET(NNBase):
    def __init__(self, drop_out=0.4, D=2, F1=96, F2=96, kernLength=256, dropoutType='Dropout',
                 fs=128, channels=2, num_classes=4):
        super().__init__(fs=fs, channels=channels, num_classes=num_classes)
        self.set_input_shape((channels, int(self.window_time * self.fs), 1))
        self.set_model_name('EEGNET')
        self.num_classes = num_classes
        self.dropoutRate = drop_out
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutType = dropoutType

    def model(self, inputs):
        if self.dropoutType == 'SpatialDropout2D':
            self.dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            self.dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same', use_bias=False)(inputs)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((inputs.shape[1], 1), use_bias=False,
                                 depth_multiplier=self.D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = self.dropoutType(self.dropoutRate)(block1)

        block2 = SeparableConv2D(self.F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = self.dropoutType(self.dropoutRate)(block2)

        flatten = Flatten(name='flatten')(block2)

        dense = Dense(self.num_classes, name='dense')(flatten)
        model = Activation('softmax', name='softmax')(dense)

        return model
