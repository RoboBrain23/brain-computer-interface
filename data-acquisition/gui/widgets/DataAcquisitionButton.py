from PySide6.QtWidgets import QPushButton, QWidget, QVBoxLayout


class StartDataAcquisitionButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setText("Start")
        self.clicked.connect(self.start_clicked)

    def start_clicked(self):
        # TODO: Implementation of Starting data acquisition process.
        print("Starting")
