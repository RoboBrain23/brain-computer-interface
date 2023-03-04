import sys

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

from widgets.LogoLabel import LogoLabel

from widgets.CountdownMessageBox import CountdownMessageBox


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

        # Description Label
        description_label = QLabel(self)
        description_label.setText("Welcome in data acquisition section")

        # Starting Button
        starting_button = QPushButton()
        starting_button.setText("Start")
        starting_button.clicked.connect(self.start_clicked)

        # Add widgets to the vertical layout
        v_layout.addWidget(logo_label)
        v_layout.addWidget(description_label)
        v_layout.addWidget(starting_button)

        self.setLayout(v_layout)

    def start_clicked(self):
        print("Countdown Started")
        self.Countdown = CountdownMessageBox()
        self.Countdown.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainDataAcquisitionWindow()
    window.show()
    app.exec()
