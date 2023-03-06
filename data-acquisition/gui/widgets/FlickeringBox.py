import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush, QPainter
from PySide6.QtWidgets import QWidget, QApplication


class FlickeringBox(QWidget):

    def __init__(self, freq: int, duration: int, fColor: str = 'black', sColor: str = 'white'):
        super().__init__()
        self._timer = None

        self.freq = freq
        self.duration = duration
        self.periodicTime = int(1000 / (2 * self.freq))  # The time of displaying one color.

        self.firstColor = fColor
        self.secondColor = sColor

        self.brushes = [QBrush(QColor(self.firstColor)), QBrush(QColor(self.secondColor))]
        self.currentColorIndex = 0
        self.counter = 0

        self.t_start = time.time()
        self.startFlashing()

    def startFlashing(self):
        self._timer = self.startTimer(self.periodicTime, Qt.PreciseTimer)

    def stopFlashing(self):
        self.currentColorIndex = 0
        if self._timer:
            self.killTimer(self._timer)
        self._timer = None
        self.update()

    def timerEvent(self, event):
        self.counter += 1
        self.currentColorIndex = (self.currentColorIndex + 1) % 2
        print(self.counter / 2)
        print(time.time() - self.t_start)
        if int(time.time() - self.t_start) == self.duration:
            print("Duration is ended!")
            self.stopFlashing()
            self.close()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.brushes[self.currentColorIndex])


if __name__ == '__main__':
    app = QApplication([])
    window = FlickeringBox(4, 2)
    # window.startFlashing()
    window.show()
    sys.exit(app.exec())
