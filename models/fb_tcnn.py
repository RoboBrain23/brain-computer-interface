from nn_base import NNBase
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Dense, Flatten, Lambda, Reshape,Add
from keras.utils import to_categorical
from keras.regularizers import l2

class FBTCNN(NNBase):
    def __init__(self,regulization_rate = 0.001,drop_out = 0.4,activation = 'elu',
                 fs = 128,channels = 2,num_classes = 4):
        """
        :param regulization_rate: the rate of the regulization

        :param drop_out: the rate of the drop_out

        :param activation: the activation function

        :param fs: the sampling frequency of the EEG data

        :param channels: the number of the channels of the EEG data

        :param num_classes: the number of the classes
        """
        self.L=l2(regulization_rate)
        self.drop_out = drop_out
        self.out_channel = 16
        self.activation = activation
        super().__init__(fs = fs,channels = channels,num_classes = num_classes,is_fb = True)
        self.set_input_shape((channels, int(self.window_time * self.fs), 3))
        self.set_model_name('FB-TCNN')

    def model(self,inputs):
        """
        :param inputs: the input data

        :return: model of the fb-tcnn
        """
        # Slice the input concatenated in data_generator to restore it to four sub-inputs
        inputs1 = self.get_input(inputs,0)
        inputs2 = self.get_input(inputs,1)
        inputs3 = self.get_input(inputs,2)
        # share the weights of the sub-inputs' three convolution layers
        conv_1 = Conv2D(self.out_channel, kernel_size=(inputs.shape[1], 1), strides=1, padding='valid',
                        kernel_regularizer=self.L)
        conv_2 = Conv2D(self.out_channel, kernel_size=(1, inputs.shape[2]), strides=5, padding='same',
                        kernel_regularizer=self.L)
        conv_3 = Conv2D(self.out_channel, kernel_size=(1, 25), strides=1, padding='valid', kernel_regularizer=self.L)
        # sub-branches
        x1 = self.sub_branch(conv_1, conv_2, conv_3, inputs1)
        x2 = self.sub_branch(conv_1, conv_2, conv_3, inputs2)
        x3 = self.sub_branch(conv_1, conv_2, conv_3, inputs3)
        # the three sub-features are fused (added)
        x = Add()([x1, x2, x3])

        # the fourth convolution layer
        x = Conv2D(32, kernel_size=(1, x.shape[2]), strides=1, padding='valid', kernel_regularizer=self.L)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Dropout(self.drop_out)(x)
        # flentten used to reduce the dimension of the features
        x = Flatten()(x)
        # the fully connected layer and "softmax"
        x = Dense(self.num_classes, activation='softmax')(x)  # shape=(None, 1, 1, 4)

        return x

    def get_input(self,inputs,i):
        """
        :param inputs: the input data

        :param i: the index of the sub-input

        :return: the sub-input
        """
        def slice_(x, index):
            return x[:, :, :, index]
        input_ = Lambda(slice_, arguments={'index': i})(inputs)
        input_ = Reshape((input_.shape[1], input_.shape[2], 1))(input_)
        return input_

    def apply_layer(self,conv, input_):
        """
        :param conv: the convolution layer

        :param input_: the input data

        :return: the output of the convolution layer
        """
        x = conv(input_)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Activation(self.activation)(x)
        x = Dropout(self.drop_out)(x)
        return x

    def sub_branch(self,conv_1, conv_2, conv_3, input_):
        """
        :param conv_1: the first shared convolution layer

        :param conv_2: the second shared convolution layer

        :param conv_3: the third shared convolution layer

        :param input_: the input data

        :return: the output of the sub-branch
        """
        # the first shared convolution layer
        x = self.apply_layer(conv_1, input_)
        # the second shared convolution layer
        x = self.apply_layer(conv_2, x)
        # the third shared convolution layer
        x = self.apply_layer(conv_3, x)
        return x
