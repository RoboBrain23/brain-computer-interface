from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          LSTM, Reshape, PReLU, Bidirectional)
from keras.models import Model
from keras.constraints import max_norm
from tensorflow_addons.layers import SpectralNormalization
from nn_base import NNBase
from keras.optimizers import Adam
import os
import utils
from keras.losses import CategoricalCrossentropy
from keras.layers import Input




class SSVEPNET(NNBase):
    # Setting hyper-parameters
    def __init__(self, drop_out=0.4, fs=128, channels=2, num_classes=4):
        super().__init__(fs=fs, channels=channels, num_classes=num_classes)
        self.num_classes = num_classes
        self.drop_out = drop_out
        self.set_model_name('SSVEPNET')

    def calculateOutSize(self,model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        # data = tf.random.normal((1, nChan, nTime, 1))
        out = model.output_shape
        return out[1:]

    def spatial_block(self,x, nChan, dropout_level):
        '''
            Spatial filter block,assign different weight to different channels and fuse them
        '''
        x = Conv2D(nChan * 2, kernel_size=(nChan, 1),kernel_constraint = max_norm(1., axis=(0,1,2)))(x)
        x = BatchNormalization()(x) 
        x = PReLU()(x)
        x = Dropout(dropout_level)(x)
        return x

    def enhanced_block(self,x, out_channels, dropout_level, kernel_size, stride):
        '''
            Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        x = Conv2D(out_channels, kernel_size=(1, kernel_size),strides=(1, stride))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_level)(x)
        return x

    def dense_layers(self,x,D1,D2,dropout_level):
        '''
            Dense layers,fully connected layers
        '''
        x = Flatten()(x)
        x = Dense(D1)(x)
        x = PReLU()(x)
        x = Dense(D2)(x)
        x = PReLU()(x)
        x = Dropout(dropout_level)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return x
    
    def apply_spectral_normalization(self,model):
        '''
            Apply spectral normalization to the model
        '''
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                spectral_layer = SpectralNormalization(layer)
                model.layers[model.layers.index(layer)] = spectral_layer
        return model

    def BiLSTM(self,x, hidden_size):
        """
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
        """
        # x = Reshape((input_size, hidden_size))(x)
        x = Bidirectional(LSTM(hidden_size, return_sequences=True))(x)
        # x = Bidirectional(LSTM(hidden_size))(x)
        # x = Attention()(x)
        x = Flatten()(x)
        return x



    def model(self,input_shape):
        '''
            Build the model
        '''
        F = [self.channels * 2] + [self.channels * 4]
        T = input_shape[1]
        K = 10
        S = 2
        inputs = Input(shape=(input_shape[0], T,1))
        net = self.spatial_block(inputs,input_shape[0], self.drop_out)
        net = self.enhanced_block(net, F[1], self.drop_out,K, S)
        conv_layers = net
        # conv_layers = Model(inputs, net)
        # conv_layers.summary()
        fcSize = self.calculateOutSize(Model(inputs, net), input_shape[0], T)
        fcUnit = fcSize[0] * fcSize[1] * fcSize[2] * 2
        D1 = fcUnit // 10
        D2 = D1 // 5
        # add Reshape layer to reduce the dimension of the output of conv_layers
        out = Reshape((fcSize[0], fcSize[1] * fcSize[2]))(conv_layers)
        rnn = self.BiLSTM(out,F[1])

        out = self.dense_layers(rnn,D1,D2,self.drop_out)
        # Model(inputs, out).summary()
        return self.apply_spectral_normalization(Model(inputs, out))


    def train(self, train_data, y_label, start_time, lr=0.01,
              batch_size=256, epochs=400, reduce_lr=True,
              check_point=False, check_point_mode='min', csv_logger=False,
              tensorboard=False, tensorboard_write_graph=True, tensorboard_histogram_freq=1,
              reducelr_patience=20, reducelr_factor=0.7, reducelr_min_lr=0.0001, reducelr_verbose=1,
              fs=128, save=False, save_path=''):
        """
        train the model

        :param train_data: the training data

        :param y_label: the label of the training data

        :param start_time: the start time of the time-window

        :param batch_size: the number of the training samples in each batch

        :param epochs: the number of the training epochs

        :param reduce_lr: whether to use the ReduceLROnPlateau callback or not

        :param check_point: whether to use the ModelCheckpoint callback or not

        :param check_point_mode: the mode of the ModelCheckpoint callback

        :param csv_logger: whether to use the CSVLogger callback or not

        :param tensorboard: whether to use the TensorBoard callback or not

        :param tensorboard_write_graph: whether to write the graph of the model in the TensorBoard callback or not

        :param tensorboard_histogram_freq: the frequency of the histogram in the TensorBoard callback

        :param reducelr_patience: the patience of the ReduceLROnPlateau callback

        :param reducelr_factor: the factor of the ReduceLROnPlateau callback

        :param reducelr_min_lr: the minimum learning rate of the ReduceLROnPlateau callback

        :param reducelr_verbose: the verbose of the ReduceLROnPlateau callback

        :param fs: the sampling frequency of the EEG data

        :param save: whether to save the model or not

        :param save_mode: the mode of the saving model

        :param save_path: the path of the saving model
        """

        if not os.path.exists(save_path) and save_path != '':
            os.makedirs(save_path)

        train_data = utils.filter_data(train_data,fs=self.fs,is_fb=self.is_fb)
        train_list, val_list = utils.train_val_split(y_label)
        train_generator = utils.train_data_generator(batch_size, train_data, y_label, start_time, train_list,
                                                     self.num_classes, self.input_shape,is_fb=self.is_fb)
        val_generator = utils.train_data_generator(batch_size, train_data, y_label, start_time, val_list,
                                                   self.num_classes, self.input_shape,is_fb=self.is_fb)
        model = self.model(self.input_shape)
        model_name = f'{self.model_name}-{str(self.window_time)}s-' + '{epoch:02d}-{val_accuracy:.4f}.h5'
        model_name = os.path.join(save_path, model_name)
        
        adam = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        callbacks = self.set_callbacks(check_point=check_point, check_point_mode=check_point_mode,
                                check_point_path=model_name, csv_logger=csv_logger,
                                tensorboard=tensorboard,
                                tensorboard_write_graph=tensorboard_write_graph,
                                tensorboard_histogram_freq=tensorboard_histogram_freq,
                                reduce_lr=reduce_lr,
                                reducelr_patience=reducelr_patience, reducelr_factor=reducelr_factor,
                                reducelr_min_lr=reducelr_min_lr,
                                reducelr_verbose=reducelr_verbose)

        # label smoothing
        catgorical_smooth = 0.9
        catgorical_crossentropy = CategoricalCrossentropy(label_smoothing=catgorical_smooth)
        model.compile(loss=catgorical_crossentropy, optimizer=adam, metrics=['accuracy'])
        # print(model.summary())
        history = model.fit(
            train_generator,
            steps_per_epoch=10,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=1,
            callbacks=callbacks,
            verbose=1
        )
        if save:
            self.save_model(model, f'{self.model_name}-{str(self._t_train)}s-{epochs:02d}', save_path)
