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
