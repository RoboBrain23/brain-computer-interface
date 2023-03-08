from PySide6.QtWidgets import QWidget, QGridLayout, QGroupBox, QRadioButton, QHBoxLayout


class FlickeringModeGroupBox(QWidget):
    """
    GroupBox which contains training and controlling mode
    """
    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        self.setLayout(layout)

        groupbox = QGroupBox("Flickering Mode")
        layout.addWidget(groupbox)

        hBox = QHBoxLayout()
        groupbox.setLayout(hBox)

        self.training = QRadioButton("Training")
        self.training.setChecked(True)
        hBox.addWidget(self.training)

        self.controlling = QRadioButton("Controlling")
        hBox.addWidget(self.controlling)

    def isTraining(self):
        return self.training.isChecked()

    def isControlling(self):
        return self.controlling.isChecked()
