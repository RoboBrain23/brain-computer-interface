import sys
import time

import pygame

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

from data_acquisition.gui.widgets.FlickringModeGroupBox import FlickeringModeGroupBox
from data_acquisition.gui.widgets.LogoLabel import LogoLabel
from data_acquisition.gui.widgets.stimulus.Stimulus import Stimulus


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Stimulus Configurations
        # TODO: Customize these configurations as you like.
        # Start of configurations

        self.stimulusScreenWidth = 1024
        self.stimulusScreenHeight = 768

        self.TOP_FREQ = 9  # HZ => 6.666666666666667
        self.RIGHT_FREQ = 8  # HZ => 7.5
        self.DOWN_FREQ = 7  # HZ => 8.571428571428571
        self.LEFT_FREQ = 6  # HZ => 10.0

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

        # Starting Button
        self.startingButton = QPushButton()
        self.startingButton.setText("Start")
        self.startingButton.clicked.connect(self.startClicked)

        # Add widgets to the vertical layout
        self.vLayout.addWidget(self.logoLabel)
        self.vLayout.addWidget(self.descriptionLabel)
        self.vLayout.addWidget(self.flickeringModeGroup)
        self.vLayout.addWidget(self.startingButton)

        self.setLayout(self.vLayout)

        # Stimulus initialization
        self.screen = None
        self.clock = None
        self.stimulus = None
        self.done = False
        self.display = [self.stimulusScreenWidth, self.stimulusScreenHeight]

    def initStimulusGUI(self, isTraining: bool):
        """
        Initialize and display the SSVEP Stimulus GUI.

        :param isTraining: bool value to give an indication to run training stimulus mode.
        """

        pygame.init()
        pygame.display.set_caption("SSVEP Stimulus")
        self.screen = pygame.display.set_mode(self.display)

        self.done = False
        self.clock = pygame.time.Clock()
        self.stimulus = Stimulus(self.screen)
        # self.stimulus = stimulus.FlickeringManager(self.screen)

        frequencies = [self.LEFT_FREQ, self.TOP_FREQ, self.RIGHT_FREQ, self.DOWN_FREQ]

        for i in range(len(frequencies)):
            self.done = False

            if isTraining:
                self.stimulus.add(self.stimulus.CENTER, frequencies[i])
            else:
                # Controlling mode is ON
                self.stimulus.add(self.stimulus.LEFT, self.LEFT_FREQ)
                self.stimulus.add(self.stimulus.TOP, self.TOP_FREQ)
                self.stimulus.add(self.stimulus.RIGHT, self.RIGHT_FREQ)
                self.stimulus.add(self.stimulus.DOWN, self.DOWN_FREQ)

            startingTime = time.time()
            while not self.done:
                if isTraining:
                    if (time.time() - startingTime) >= self.EPOC_DURATION:
                        if i < len(frequencies) - 1:
                            pygame.time.wait(self.BREAK_DURATION)
                        break

                for event in pygame.event.get():
                    if (event.type == pygame.KEYUP) or (event.type == pygame.KEYDOWN):
                        if event.key == pygame.K_ESCAPE:
                            self.done = True
                    if event.type == pygame.QUIT:
                        self.done = True

                self.screen.fill((0, 0, 0))
                self.clock.tick(60)  # 16 ms between frames ~ 60FPS
                self.stimulus.process()
                self.stimulus.draw()
                pygame.display.flip()

            if not isTraining or self.done:
                break
        pygame.quit()

    def startClicked(self):
        """
        Click listener for the starting button
        """
        self.initStimulusGUI(self.flickeringModeGroup.isTraining())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
