import sys

import pygame
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QRadioButton, \
    QHBoxLayout

from widgets.LogoLabel import LogoLabel

from widgets.CountdownMessageBox import CountdownMessageBox

from widgets.FlickeringBox import FlickeringManager
from widgets.FlickringModeGroupBox import FlickeringModeGroupBox


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.Countdown = None
        self.setWindowTitle("Data Acquisition")
        v_layout = QVBoxLayout()

        # Logo Label
        logo_width = 400
        logo_height = 400
        logo_label = LogoLabel(logo_width, logo_height)
        logo_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Description Label
        description_label = QLabel()
        description_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        description_label.setText("Welcome in data acquisition section")

        # TODO: Implement RadioButton for choosing training with epocing duration or production
        self.flickeringModeGroup = FlickeringModeGroupBox()



        # TODO: Implement frequencies options

        # Starting Button
        starting_button = QPushButton()
        starting_button.setText("Start")
        starting_button.clicked.connect(self.start_clicked)

        # Add widgets to the vertical layout
        v_layout.addWidget(logo_label)
        v_layout.addWidget(description_label)
        v_layout.addWidget(self.flickeringModeGroup)
        # v_layout.addWidget(training)
        # v_layout.addWidget(production)
        v_layout.addWidget(starting_button)

        self.setLayout(v_layout)

        # Flickering box initialization
        self.screen = None
        self.clock = None
        self.flickering = None
        self.done = False
        self.display = [1024, 768]

    def startFlickering(self):
        """
        Start the flickering boxes GUI.
        """
        pygame.init()
        pygame.display.set_caption("Flickering Box")
        self.screen = pygame.display.set_mode(self.display)

        self.done = False
        self.clock = pygame.time.Clock()
        self.flickering = FlickeringManager(self.screen)

        self.flickering.add(self.flickering.LEFT, 6)  # frame: 6, HZ => 10.0
        self.flickering.add(self.flickering.TOP, 4)  # frame: 4, HZ => 15.0
        self.flickering.add(self.flickering.RIGHT, 5)  # frame: 5, HZ => 12.0
        self.flickering.add(self.flickering.DOWN, 8)  # frame: 8, HZ => 7.5
        self.flickering.add(self.flickering.CENTER, 10)  # frame: 10, HZ => 6.0

        while not self.done:
            for event in pygame.event.get():
                if (event.type == pygame.KEYUP) or (event.type == pygame.KEYDOWN):
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                if event.type == pygame.QUIT:
                    self.done = True

            self.screen.fill((0, 0, 0))
            self.clock.tick(60)  # 16 ms between frames ~ 60FPS
            self.flickering.process()
            self.flickering.draw()
            pygame.display.flip()

        pygame.quit()

    def start_clicked(self):
        """
        Click listener for the starting button
        """
        print("Countdown Started")
        print(self.flickeringModeGroup.isTraining())
        print(self.flickeringModeGroup.isControlling())
        # self.Countdown = CountdownMessageBox(2)
        # self.Countdown.show()
        # self.startFlickering()

        # main.main()
        # self.f = FlickeringBox(20, 5)
        # self.f.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # window = FlickeringBox(15)
    # window.show()
    app.exec()

    # # Print frames with the corresponding HZ
    # for i in range(1, 61):
    #     print("frame: {f}, HZ => {hz}".format(f=i, hz=60 / i))
