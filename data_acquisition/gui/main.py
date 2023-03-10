import sys
import time

import pygame

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox

from data_acquisition.gui.widgets.FlickringModeGroupBox import FlickeringModeGroupBox
from data_acquisition.gui.widgets.LogoLabel import LogoLabel
from data_acquisition.gui.widgets.stimulus.Stimulus import Stimulus


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Stimulus Configurations
        # TODO: Customize these configurations as you like.
        # Start of configurations

        self.isFullScreen = False
        self.stimulusScreenWidth = 1024
        self.stimulusScreenHeight = 768

        self.TOP_FREQ = 4  # HZ => 15
        self.RIGHT_FREQ = 5  # HZ => 12
        self.DOWN_FREQ = 7  # HZ => 8.571428571428571
        self.LEFT_FREQ = 6  # HZ => 10.0

        self.frequencies = [self.TOP_FREQ, self.RIGHT_FREQ, self.DOWN_FREQ, self.LEFT_FREQ]

        self.EPOC_DURATION = 2  # Seconds
        self.BREAK_DURATION = 2000  # Milliseconds

        # End of configurations

        self.setWindowTitle("SSVEP Stimulus")
        self.vLayout = QVBoxLayout()

        # GUI Components
        self.Countdown = None
        self.stimulus = None

        # Logo Label
        self._logoWidth = 400
        self._logoHeight = 400
        self.logoLabel = LogoLabel(self._logoWidth, self._logoHeight)
        self.logoLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Description Label
        self.descriptionLabel = QLabel()
        self.descriptionLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.descriptionLabel.setText("Welcome in data acquisition section")

        # GroupBox of RadioButton for training or controlling mode
        self.flickeringModeGroup = FlickeringModeGroupBox()

        # CheckBox for full-screen mode
        self.fullScreenCheckBox = QCheckBox("Full Screen mode")
        self.fullScreenCheckBox.setChecked(self.isFullScreen)

        # Starting Button
        self.startingButton = QPushButton()
        self.startingButton.setText("Start")
        self.startingButton.clicked.connect(self.startClicked)

        # Add widgets to the vertical layout
        self.vLayout.addWidget(self.logoLabel)
        self.vLayout.addWidget(self.descriptionLabel)
        self.vLayout.addWidget(self.flickeringModeGroup)
        self.vLayout.addWidget(self.fullScreenCheckBox)
        self.vLayout.addWidget(self.startingButton)

        self.setLayout(self.vLayout)

    def startClicked(self):
        """
        Click listener for the starting button
        """
        self.isFullScreen = self.fullScreenCheckBox.isChecked()
        self.stimulus = Stimulus(self.isFullScreen, self.stimulusScreenWidth, self.stimulusScreenHeight, self.frequencies,
                                 self.EPOC_DURATION, self.BREAK_DURATION)
        self.stimulus.run(self.flickeringModeGroup.isTraining())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
