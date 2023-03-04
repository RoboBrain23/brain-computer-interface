from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox


class CountdownMessageBox(QMessageBox):
    def __init__(self, timeout=5):
        super().__init__()
        self.setWindowTitle("Be Ready")
        self.time_to_wait = timeout
        self.setText("The Session will start after {0} seconds.".format(timeout))
        self.setStandardButtons(QMessageBox.NoButton)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.onChangeContent)
        self.timer.start()

    def onChangeContent(self):
        if self.time_to_wait > 0:
            self.time_to_wait -= 1
            self.setText("The Session will start after {0} seconds.".format(self.time_to_wait))
            print(self.time_to_wait)
        else:
            self.close()

    def closeEvent(self, event):
        print("Countdown Ended")
        self.timer.stop()
        event.accept()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = CountdownMessageBox(3)
#     ex.show()
#     app.exec()
#     # sys.exit(app.exec())
