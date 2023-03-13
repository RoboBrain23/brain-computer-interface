import asyncio
import sys
import threading

from PySide6.QtWidgets import QApplication
from data_acquisition.gui.MainWindow import MainWindow
from data_acquisition.modules.utils.stimulus.Stimulus import Stimulus

from data_acquisition.modules.headset.EEG import EEG


# epoc_headset = EEG()
# epoc_headset.get_headset_info()
# epoc_headset.record_csv()
#
#
#


# import asyncio
# import threading
#
# async def record_eeg_csv():
#
#     epoc_headset = EEG()
#     epoc_headset.get_headset_info()
#     epoc_headset.record_csv()
#
# async def run_gui():
#     app = QApplication(sys.argv)
#     window = SSVEP()
#     window.show()
#     app.exec()
#
#
# def run_ssvep():
#
#     loop = asyncio.get_event_loop()
#     try:
#         asyncio.ensure_future(run_gui())
#         asyncio.ensure_future(record_eeg_csv())
#         loop.run_forever()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         print("Closing Loop")
#         loop.close()


# # Function to run in asyncio event loop
# async def record_async():
#     loop = asyncio.get_running_loop()
#     # Use run_in_executor to run blocking function in separate thread
#     await loop.run_in_executor(None, record_eeg_csv)


# Function to start asyncio event loop and run the GUI
# def start():
#     # Start asyncio event loop in separate thread
#     asyncio_thread = threading.Thread(target=asyncio.run, args=(record_async(),))
#     asyncio_thread.start()
#
#     # Run GUI event loop in main thread
#     # ...
#     app = QApplication(sys.argv)
#     window = SSVEP()
#     window.show()
#     app.exec()
#
# start()


class SSVEP(MainWindow):
    def __init__(self):
        super().__init__()
        self.epoc_headset = None

    def startGUI(self):
        fullScreen = self.isFullScreen()
        self.stimulus = Stimulus(fullScreen, self.stimulusScreenWidth, self.stimulusScreenHeight, self.frequencies,
                                 self.EPOC_DURATION, self.BREAK_DURATION)
        self.stimulus.run(self.flickeringModeGroup.isTraining())

    async def startRecording(self):
        self.epoc_headset = EEG()
        self.epoc_headset.get_headset_info()
        self.epoc_headset.record_csv()

    async def startRecordingAsync(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.startRecording)

    def run(self):
        asyncio_thread = threading.Thread(target=asyncio.run, args=(self.startRecording(),))
        asyncio_thread.start()
        self.startGUI()

    def startClicked(self):
        self.run()






app = QApplication(sys.argv)
window = SSVEP()
window.show()
app.exec()

# run_ssvep()
