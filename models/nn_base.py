import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model, load_model


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

    def __train_val_split(self, y_label):
        train_list = np.arange(y_label.shape[0])
        val_list = np.random.choice(train_list, size=y_label.shape[0] // 10, replace=False)
        train_list = np.delete(train_list, val_list)
        return train_list, val_list

    def train(self, train_data, y_label, start_time, lr=0.01,
              batch_size=256, epochs=400, reduce_lr=True,
              check_point=False, check_point_mode='min', csv_logger=False,
              tensorboard=False, tensorboard_write_graph=True, tensorboard_histogram_freq=1,
              reducelr_patience=20, reducelr_factor=0.7, reducelr_min_lr=0.0001, reducelr_verbose=1,
              fs=128, save_path='model'):
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
        # model_name = 'tCnn_model_'+str(t_train)+'s.h5'
        model_name = save_path + '_tCNN_' + str(self.window_time) + 's.{epoch:02d}-{val_accuracy:.4f}.h5'
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        callbacks = self.set_callbacks(check_point=check_point, check_point_mode=check_point_mode,
                                       check_point_path=model_name, csv_logger=csv_logger, tensorboard=tensorboard,
                                       tensorboard_write_graph=tensorboard_write_graph,
                                       tensorboard_histogram_freq=tensorboard_histogram_freq,
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
            callbacks=callbacks
        )

    def test(self, test_data, y_label, start_time, batch_size=32, fs=128, model_path='model'):
        win_train = int(self.window_time * fs)
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
        self.input_shape = input_shape

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_window_time(self, window_time):
        self.window_time = window_time
        self.input_shape = (self.channels, int(self.window_time * self.fs), 1)

    def set_fs(self, fs):
        self.fs = fs
        self.input_shape = (self.channels, int(self.window_time * self.fs), 1)
