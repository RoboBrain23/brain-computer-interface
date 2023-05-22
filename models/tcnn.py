from keras.regularizers import l2
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, Dropout, Flatten
from nn_base import NNBase


class TCNN(NNBase):
    # Setting hyper-parameters
    def __init__(self, regularization_rate=0.001, drop_out=0.4, activation='elu',
                 fs=128, channels=2, num_classes=4):
        super().__init__(fs=fs, channels=channels, num_classes=num_classes)
        self.L = l2(regularization_rate)
        self.drop_out = drop_out
        self.out_channel = 16
        self.activation = activation
        self.set_model_name('TCNN')

    # the network of the tCNN
    def model(self, inputs):
        # the first convolution layer
        first = Conv2D(self.out_channel, kernel_size=(inputs.shape[1], 1), strides=1, padding='valid',
                       kernel_regularizer=self.L)(inputs)
        first = BatchNormalization()(first)
        first = Activation(self.activation)(first)
        first = Dropout(self.drop_out)(first)
        # the second convolution layer
        second = Conv2D(self.out_channel, kernel_size=(1, inputs.shape[2]), strides=5, padding='same',
                        kernel_regularizer=self.L)(first)
        second = BatchNormalization()(second)
        second = Activation(self.activation)(second)
        second = Dropout(self.drop_out)(second)
        # the third convolution layer
        third = Conv2D(self.out_channel, kernel_size=(1, 25), strides=1, padding='valid', kernel_regularizer=self.L)(
            second)
        third = BatchNormalization()(third)
        third = Activation(self.activation)(third)
        third = Dropout(self.drop_out)(third)
        # the fourth convolution layer
        fourth = Conv2D(32, kernel_size=(1, third.shape[2]), strides=1, padding='valid', kernel_regularizer=self.L)(
            third)
        fourth = BatchNormalization()(fourth)
        fourth = Activation(self.activation)(fourth)
        # flatten used to reduce the dimension of the features
        flatten = Flatten()(fourth)
        flatten = Dropout(self.drop_out)(flatten)
        # the fully connected layer and "softmax"
        out = Dense(self.num_classes, activation='softmax')(flatten)

        return out
