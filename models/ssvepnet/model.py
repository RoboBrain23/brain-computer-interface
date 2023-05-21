from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, LSTM, Reshape, PReLU, Bidirectional)
from keras.models import Model
from keras.constraints import max_norm
from tensorflow_addons.layers import SpectralNormalization


class SSVEPNET:
    # Setting hyper-parameters
    def __init__(self,  drop_out=0.4, num_classes=4, channel=2):
        self.num_classes = num_classes
        self.drop_out = drop_out
        self.channel = channel

    def calculateOutSize(self, model):
        """
        Calculate the output size of the model

        :param model: the model to be calculated

        :return: the output size of the model
        """
        out = model.output_shape
        return out[1:]

    def spatial_block(self, x, nChan, dropout_level):
        """
        Spatial block,build a CNN block to extract spatial features

        :param x: the previous layer to be connected

        :param nChan:  the number of channels

        :param dropout_level:  the dropout level

        :return:  the output of spatial block
        """
        x = Conv2D(nChan * 2, kernel_size=(nChan, 1), kernel_constraint=max_norm(1., axis=(0, 1, 2)))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_level)(x)
        return x

    def enhanced_block(self, x, out_channels, dropout_level, kernel_size, stride):
        """
        Enhanced structure block,build a CNN block to absorb data and output its stable feature

        :param x: the previous layer to be connected

        :param out_channels:  the number of output channels

        :param dropout_level:  the dropout level

        :param kernel_size:  the size of kernel

        :param stride:  the size of stride

        :return:  the output of enhanced structure block
        """
        x = Conv2D(out_channels, kernel_size=(1, kernel_size), strides=(1, stride))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_level)(x)
        return x

    def dense_layers(self, x, D1, D2, dropout_level):
        """
        Dense layers

        :param x:  the previous layer to be connected

        :param D1:  the number of neurons in the first dense layer

        :param D2:  the number of neurons in the second dense layer

        :param dropout_level:  the dropout level

        :return:  the output of dense layers
        """
        x = Flatten()(x)
        x = Dense(D1)(x)
        x = PReLU()(x)
        x = Dense(D2)(x)
        x = PReLU()(x)
        x = Dropout(dropout_level)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return x

    def apply_spectral_normalization(self, model):
        """
        Apply spectral normalization to the model to stabilize the training process
         and improve the performance of the model.

        :param model: the model to be applied spectral normalization

        :return:  the model after applying spectral normalization
        """
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                spectral_layer = SpectralNormalization(layer)
                model.layers[model.layers.index(layer)] = spectral_layer
        return model

    def BiLSTM(self, x, hidden_size):
        """
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features

        :param x: the previous layer to be connected

        :param hidden_size: the size of hidden layer

        :return: the output of Bi-LSTM layer (Flatten)
        """
        x = Bidirectional(LSTM(hidden_size))(x)
        x = Flatten()(x)
        return x

    def build_model(self, input_shape):
        """
        Build the SSVEPNet model

        :param input_shape: the shape of input data

        :return: the model of SSVEPNet
        """
        F = [self.channel * 2] + [self.channel * 4]
        T = input_shape[1]
        K = 10
        S = 2
        inputs = Input(shape=(self.channel, T, 1))
        net = self.spatial_block(inputs, self.channel, self.drop_out)
        net = self.enhanced_block(net, F[1], self.drop_out, K, S)
        conv_layers = net
        # conv_layers = Model(inputs, net)
        # conv_layers.summary()
        fcSize = self.calculateOutSize(Model(inputs, net))
        fcUnit = fcSize[0] * fcSize[1] * fcSize[2] * 2
        D1 = fcUnit // 10
        D2 = D1 // 5
        # add Reshape layer to reduce the dimension of the output of conv_layers
        out = Reshape((fcSize[0], fcSize[1] * fcSize[2]))(conv_layers)
        rnn = self.BiLSTM(out, F[1])

        out = self.dense_layers(rnn, D1, D2, self.drop_out)
        # Model(inputs, out).summary()
        return self.apply_spectral_normalization(Model(inputs, out))
