import numpy as np
from keras.utils import to_categorical


class NNBase:
    def __init__(self, fs=128, channels=2, num_classes=4):
        self.num_classes = num_classes
        self.window_time = 1
        self.model_name = 'model'
        self.fs = fs
        self.input_shape = (channels, int(self.window_time * self.fs), 1)
        self.channels = channels

    def model(self, inputs):
        pass

    def __train_data_generator(self, batch_size, train_data, win_train, y_label, start_time, train_list, channel):
        while True:
            x_train = np.zeros((batch_size, self.channel, win_train, 1), dtype=np.float32)
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

                x_train[i] = np.reshape(x_train_data, (self.channel, win_train, 1))
                y_train[i] = to_categorical(y_data, self.num_classes)
            # print(x_train.shape)
            yield x_train, y_train

    def __test_data_generator(self, batch_size, train_data, win_train, y_label, start_time, channel):
        x_train = np.zeros((batch_size, self.channel, win_train, 1), dtype=np.float32)
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

            x_train[i] = np.reshape(x_train_data, (self.channel, win_train, 1))
            y_train[i] = to_categorical(y_data, self.num_classes)

        return x_train, y_train

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_window_time(self, window_time):
        self.window_time = window_time
        self.input_shape = (self.channels, int(self.window_time * self.fs), 1)

    def set_fs(self, fs):
        self.fs = fs
        self.input_shape = (self.channels, int(self.window_time * self.fs), 1)
