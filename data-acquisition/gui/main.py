import sys

import pygame
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

from widgets.LogoLabel import LogoLabel


from widgets.CountdownMessageBox import CountdownMessageBox

from widgets.FlickeringBox import FlickeringManager


class MainDataAcquisitionWindow(QWidget):
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
        # TODO: Implement frequencies options

        # Starting Button
        starting_button = QPushButton()
        starting_button.setText("Start")
        starting_button.clicked.connect(self.start_clicked)

        # Add widgets to the vertical layout
        v_layout.addWidget(logo_label)
        v_layout.addWidget(description_label)
        v_layout.addWidget(starting_button)

        self.setLayout(v_layout)

        # Flickering box initialization
        self.screen = None
        self.clock = None
        self.flickeringManager = None
        self.done = False
        self.display = [1024, 768]


    def startFlickering(self):
        pygame.init()
        pygame.display.set_caption("Flickering Box")
        self.screen = pygame.display.set_mode(self.display)

        self.done = False
        self.clock = pygame.time.Clock()
        self.flickeringManager = FlickeringManager(self.screen)

        self.flickeringManager.add('left', 1)
        self.flickeringManager.add('top', 5)
        self.flickeringManager.add('right', 6)
        self.flickeringManager.add('bottom', 7)

        while not self.done:
            for event in pygame.event.get():
                if (event.type == pygame.KEYUP) or (event.type == pygame.KEYDOWN):
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                if event.type == pygame.QUIT:
                    self.done = True
            self.screen.fill((0, 0, 0))
            self.clock.tick(60)  # 16 ms between frames ~ 60FPS
            self.flickeringManager.process()
            self.flickeringManager.draw()
            pygame.display.flip()

        pygame.quit()



    def start_clicked(self):
        # print("Countdown Started")
        # self.Countdown = CountdownMessageBox(2)
        # self.Countdown.show()
        self.startFlickering()


        # main.main()
        # self.f = FlickeringBox(20, 5)
        # self.f.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainDataAcquisitionWindow()
    window.show()
    # window = FlickeringBox(15)
    # window.show()
    app.exec()
