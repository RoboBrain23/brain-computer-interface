import sys

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

from widgets.DataAcquisitionButton import StartDataAcquisitionButton
from widgets.LogoLabel import LogoLabel

app = QApplication(sys.argv)


class MainDataAcquisitionWindow(QWidget):
    def __init__(self):
        super().__init__()

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
        starting_button = StartDataAcquisitionButton()

        # Add widgets to the vertical layout
        v_layout.addWidget(logo_label)
        v_layout.addWidget(description_label)
        v_layout.addWidget(starting_button)

        self.setLayout(v_layout)


window = MainDataAcquisitionWindow()

window.show()

app.exec()
