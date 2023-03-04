from PySide6 import QtGui, QtCore
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel









class LogoLabel(QLabel):
    def __init__(self, logo_width, logo_height):
        super().__init__()

        logo_size = QSize(logo_width, logo_height)

        logo = QtGui.QPixmap('resources/images/logo.png')
        logo = logo.scaled(logo_size, QtCore.Qt.KeepAspectRatio)

        self.setPixmap(logo)
