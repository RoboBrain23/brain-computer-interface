import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model, load_model
import os


class NNBase:
    def __init__(self, fs=128, channels=2, num_classes=4):
        """
        NN is the base class of all the neural networks used in this project

        :param fs: the sampling frequency of the EEG data

        :param channels: the number of the channels of the EEG data

        :param num_classes: the number of the classes
        """
        self.num_classes = num_classes
        self.window_time = 1
        self.model_name = 'model'
        self.fs = fs
        self.input_shape = (channels, int(self.window_time * self.fs), 1)
        self.channels = channels

    def model(self, inputs):
        """
        the model of the neural network

        :param inputs: the input of the neural network
        """
        pass

    def __train_data_generator(self, batch_size, train_data, win_train, y_label, start_time, train_list, channel):
        """
        the generator of the training data

        :param batch_size:  the size of the batch

        :param train_data:  the training data

        :param win_train:  the length of the training data

        :param y_label:  the label of the training data

        :param start_time:  the start time of the training data

        :param train_list:  the list of the training data

        :param channel:  the number of the channels

        :return:  the training data and the label of the training data
        """
        while True:
            x_train = np.zeros((batch_size, self.channels, win_train, 1), dtype=np.float32)
            y_train = np.zeros((batch_size, self.num_classes), dtype=np.float32)
            # get training samples of batch_size trials
            for i in range(batch_size):
                # randomly selecting the single-trial
                k = np.random.choice(train_list)
                # get the label of the single-trial
                y_data = y_label[k]
                # randomly selecting a single-sample in the single-trial,630 frames is the data range we used
                # 630 frames = 4.9s * 128Hz
                time_start = np.random.randint(0, int(630 - win_train)) - start_time[0]
                x1 = int(start_time[k]) + time_start
                x2 = int(start_time[k]) + time_start + win_train
                x_train_data = train_data[x1:x2, :]
                # todo: add preprocessing functions here

                x_train[i] = np.reshape(x_train_data, (self.channels, win_train, 1))
                y_train[i] = to_categorical(y_data, self.num_classes)
            # print(x_train.shape)
            yield x_train, y_train

    def __test_data_generator(self, batch_size, train_data, win_train, y_label, start_time, channel):
        """
        the generator of the testing data

        :param batch_size:  the size of the batch

        :param train_data:  the training data

        :param win_train:  the length of the training data

        :param y_label:  the label of the training data

        :param start_time:  the start time of the training data

        :param channel:  the number of the channels

        :return:  the training data and the label of the training data
        """
        x_train = np.zeros((batch_size, self.channels, win_train, 1), dtype=np.float32)
        y_train = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        # get training samples of batch_size trials
        for i in range(batch_size):
            # randomly selecting the single-trial
            k = np.random.randint(0, (y_label.shape[0] - 1))
            # get the label of the single-trial
            y_data = y_label[k]
            # randomly selecting a single-sample in the single-trial,630 frames is the data range we used
            # 630 frames = 4.9s * 128Hz
            time_start = np.random.randint(0, int(630 - win_train)) - start_time[0]
            x1 = int(start_time[k]) + time_start
            x2 = int(start_time[k]) + time_start + win_train
            x_train_data = train_data[x1:x2, :]
            # todo: add preprocessing functions here

            x_train[i] = np.reshape(x_train_data, (self.channels, win_train, 1))
            y_train[i] = to_categorical(y_data, self.num_classes)

        return x_train, y_train

    def __train_val_split(self, y_label):
        """
        split the training data into training set and validation set

        :param y_label:  the label of the training data

        :return:  the index of the training set and validation set
        """
        train_list = np.arange(y_label.shape[0])
        val_list = np.random.choice(train_list, size=y_label.shape[0] // 10, replace=False)
        train_list = np.delete(train_list, val_list)
        return train_list, val_list

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

        win_trian = int(self.window_time * self.fs)
        train_list, val_list = self.__train_val_split(y_label)
        train_generator = self.__train_data_generator(batch_size, train_data, win_trian, y_label, start_time,
                                                      train_list, train_data.shape[0])
        val_generator = self.__train_data_generator(batch_size, train_data, win_trian, y_label, start_time,
                                                    val_list, train_data.shape[0])
        input_tensor = Input(shape=self.input_shape)
        preds = self.model(input_tensor)
        model = Model(input_tensor, preds)
        # model.summary()
        model_name = f'{self.model_name}-{str(self.window_time)}s-' + '{epoch:02d}-{val_accuracy:.4f}.h5'
        model_name = os.path.join(save_path, model_name)
        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        callbacks = self.set_callbacks(check_point=check_point, check_point_mode=check_point_mode,
                                       check_point_path=model_name, csv_logger=csv_logger,
                                       tensorboard=tensorboard,
                                       tensorboard_write_graph=tensorboard_write_graph,
                                       tensorboard_histogram_freq=tensorboard_histogram_freq,
                                       reduce_lr=reduce_lr,
                                       reducelr_patience=reducelr_patience, reducelr_factor=reducelr_factor,
                                       reducelr_min_lr=reducelr_min_lr,
                                       reducelr_verbose=reducelr_verbose)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
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

    def test(self, test_data, y_label, start_time, batch_size=32, fs=128, model_path='model'):
        """
        test the model with the test data and return the accuracy

        :param test_data: the test data

        :param y_label: the label of the test data

        :param start_time: the start time of the test data

        :param model_path: the path of the model

        :param batch_size: the batch size of the test data

        :param fs: the sampling frequency

        :return: the accuracy of the test data
        """
        if model_path == 'model':
            model_name = model_path + '_tCNN_' + str(self.window_time) + 's.{epoch:02d}-{val_accuracy:.4f}.h5'
        else:
            model_name = model_path
        model = load_model(model_name)
        print('load model: ', model_name)
        acc_list = []
        av_acc_list = []
        for i in range(5):
            x_train, y_train = self.__test_data_generator(batch_size, test_data, win_trian, y_label, start_time,
                                                          test_data.shape[0])
            a, b = 0, 0
            # get the predicted results of the batchsize test samples
            y_pred = model.predict(np.array(x_train))
            true, pred = [], []
            y_true = y_train
            # Calculating the accuracy of current time
            for i in range(batch_size - 1):
                y_pred_ = np.argmax(y_pred[i])
                pred.append(y_pred_)
                y_true1 = np.argmax(y_train[i])
                true.append(y_true1)
                if y_true1 == y_pred_:
                    a += 1
                else:
                    b += 1
            acc = a / (a + b)
            acc_list.append(acc)
        av_acc = np.mean(acc_list)
        print(av_acc)

    def set_callbacks(self, check_point=True, check_point_mode='min', check_point_path='', csv_logger=True,
                      tensorboard=True, tensorboard_write_graph=True, tensorboard_histogram_freq=1, reduce_lr=True,
                      reducelr_patience=20, reducelr_factor=0.7, reducelr_min_lr=0.0001, reducelr_verbose=1):
        """
        set the callbacks of the model

        :param check_point: whether to use the ModelCheckpoint callback

        :param check_point_mode: the mode of the ModelCheckpoint callback

        :param check_point_path: the path of the ModelCheckpoint callback

        :param csv_logger: whether to use the CSVLogger callback

        :param tensorboard: whether to use the TensorBoard callback

        :param tensorboard_write_graph: whether to write the graph of the TensorBoard callback

        :param tensorboard_histogram_freq: the frequency of the TensorBoard callback

        :param reduce_lr: whether to use the ReduceLROnPlateau callback

        :param reducelr_patience: the patience of the ReduceLROnPlateau callback

        :param reducelr_factor: the factor of the ReduceLROnPlateau callback

        :param reducelr_min_lr: the min_lr of the ReduceLROnPlateau callback

        :param reducelr_verbose: the verbose of the ReduceLROnPlateau callback
        """
        callbacks = []
        if reduce_lr:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reducelr_factor,
                                          patience=reducelr_patience, min_lr=reducelr_min_lr, verbose=reducelr_verbose)
            callbacks.append(reduce_lr)
        if check_point:
            model_checkpoint = ModelCheckpoint(check_point_path, monitor='val_accuracy', verbose=1,
                                               save_best_only=True, mode=check_point_mode)
            callbacks.append(model_checkpoint)
        if csv_logger:
            csv_logger = CSVLogger(self.model_name + ".log")
            callbacks.append(csv_logger)
        if tensorboard:
            tensorboard = TensorBoard(log_dir=self.model_name, write_graph=tensorboard_write_graph,
                                      histogram_freq=tensorboard_histogram_freq)
            callbacks.append(tensorboard)

        return callbacks

    def set_input_shape(self, input_shape):
        """
        set the input shape of the model

        :param input_shape: the input shape of the model
        """
        self.input_shape = input_shape

    def set_model_name(self, model_name):
        """
        set the name of the model

        :param model_name: the name of the model
        """
        self.model_name = model_name

    def set_window_time(self, window_time):
        """
        set the window time that used in training

        :param window_time: the window time that used in training
        """
        self.window_time = window_time
        self.input_shape = (self.channels, int(self.window_time * self.fs), 1)

    def set_fs(self, fs):
        """
        set the sampling frequency of the data

        :param fs: the sampling frequency of the data
        """
        self.fs = fs
        self.input_shape = (self.channels, int(self.window_time * self.fs), 1)

    def save_model(self, model, model_name, save_path=''):
        """
        save the model to the save_path

        :param model: the model that need to be saved

        :param model_name: the name of the model

        :param save_path: the path of the model
        """
        model_name = os.path.join(save_path, model_name)
        model_name = model_name + '.h5'
        print(model_name)
        model.save(model_name)
